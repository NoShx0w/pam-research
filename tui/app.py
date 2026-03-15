from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable

import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header

from tui.widgets.panel import Panel
from tui.widgets.detail_view import DetailSelection, DetailView


INDEX_PATH = Path("outputs/index.csv")
SWEEP_SPEC_PATH = Path("tui/sweep_spec.json")
REFRESH_SECONDS = 5.0
DEFAULT_SEEDS_PER_CELL = 10
COLUMN_WIDTH = 7


@dataclass
class SweepSpec:
    r_values: list[float]
    alpha_values: list[float]
    seeds_per_cell: int = DEFAULT_SEEDS_PER_CELL

    @property
    def expected_total(self) -> int:
        return len(self.r_values) * len(self.alpha_values) * self.seeds_per_cell


@dataclass
class Snapshot:
    df: pd.DataFrame
    row_count: int
    completed: int
    expected_total: int
    percent: float
    observed_r: int
    observed_alpha: int
    last_modified: str
    sweep_spec: SweepSpec


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sorted_unique_numeric(series: pd.Series) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    return sorted(float(v) for v in vals)


def display_float(x: float, digits: int = 3) -> str:
    s = f"{float(x):.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_field(label: str, value: object, width: int = 12) -> str:
    return f"{label:<{width}} {value}"


def _load_sweep_spec_file(path: Path) -> SweepSpec | None:
    if not path.exists():
        return None
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    r_values = spec.get("r_values") or spec.get("r") or spec.get("rs")
    alpha_values = spec.get("alpha_values") or spec.get("alpha") or spec.get("alphas")
    seeds_per_cell = spec.get("seeds_per_cell", spec.get("seeds", DEFAULT_SEEDS_PER_CELL))

    if not isinstance(r_values, list) or not isinstance(alpha_values, list):
        return None

    try:
        return SweepSpec(
            r_values=sorted(float(x) for x in r_values),
            alpha_values=sorted(float(x) for x in alpha_values),
            seeds_per_cell=int(seeds_per_cell),
        )
    except Exception:
        return None


def _fallback_sweep_spec(df: pd.DataFrame) -> SweepSpec:
    work = _safe_numeric(df, ["r", "alpha"]) if not df.empty else pd.DataFrame()
    if not work.empty and {"r", "alpha"}.issubset(work.columns):
        r_values = _sorted_unique_numeric(work["r"])
        alpha_values = _sorted_unique_numeric(work["alpha"])
        if r_values and alpha_values:
            return SweepSpec(r_values=r_values, alpha_values=alpha_values, seeds_per_cell=DEFAULT_SEEDS_PER_CELL)

    return SweepSpec(
        r_values=[0.10, 0.15, 0.20, 0.25, 0.30],
        alpha_values=[0.03, 0.039, 0.047, 0.056, 0.064, 0.073, 0.081, 0.09, 0.099, 0.107, 0.116, 0.124, 0.133, 0.141, 0.15],
        seeds_per_cell=DEFAULT_SEEDS_PER_CELL,
    )


def load_sweep_spec(df: pd.DataFrame) -> SweepSpec:
    explicit = _load_sweep_spec_file(SWEEP_SPEC_PATH)
    return explicit if explicit is not None else _fallback_sweep_spec(df)


def load_snapshot(index_path: Path) -> Snapshot:
    if not index_path.exists():
        empty = pd.DataFrame()
        spec = load_sweep_spec(empty)
        return Snapshot(
            df=empty,
            row_count=0,
            completed=0,
            expected_total=spec.expected_total,
            percent=0.0,
            observed_r=0,
            observed_alpha=0,
            last_modified="missing",
            sweep_spec=spec,
        )

    try:
        df = pd.read_csv(index_path)
    except Exception:
        df = pd.DataFrame()

    spec = load_sweep_spec(df)
    completed = len(df)
    percent = 100.0 * completed / spec.expected_total if spec.expected_total else 0.0

    if not df.empty and {"r", "alpha"}.issubset(df.columns):
        work = _safe_numeric(df, ["r", "alpha"])
        observed_r = int(work["r"].dropna().nunique())
        observed_alpha = int(work["alpha"].dropna().nunique())
    else:
        observed_r = 0
        observed_alpha = 0

    try:
        last_modified = pd.Timestamp(index_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        last_modified = "unknown"

    return Snapshot(
        df=df,
        row_count=len(df),
        completed=completed,
        expected_total=spec.expected_total,
        percent=percent,
        observed_r=observed_r,
        observed_alpha=observed_alpha,
        last_modified=last_modified,
        sweep_spec=spec,
    )


def build_status_text(snap: Snapshot, selection: DetailSelection, qph: float) -> str:
    lines = [
        _fmt_field("index path", INDEX_PATH),
        _fmt_field("rows loaded", snap.row_count),
        _fmt_field("completed", f"{snap.completed} / {snap.expected_total}"),
        _fmt_field("progress", f"{snap.percent:6.2f}%"),
        _fmt_field("throughput", f"{qph:8.2f} q/h"),
        _fmt_field("observed r", f"{snap.observed_r} / {len(snap.sweep_spec.r_values)}"),
        _fmt_field("observed α", f"{snap.observed_alpha} / {len(snap.sweep_spec.alpha_values)}"),
        _fmt_field("mode", selection.mode),
        _fmt_field("selected r", display_float(selection.selected_r, 3) if selection.selected_r is not None else "—"),
        _fmt_field("selected α", display_float(selection.selected_alpha, 3) if selection.selected_alpha is not None else "—"),
        _fmt_field("controls", "↑↓ r   ←→ α"),
        _fmt_field("enter", "row/cell toggle"),
        _fmt_field("t", "trajectory mode"),
        _fmt_field("last modified", snap.last_modified),
        _fmt_field("refresh every", f"{REFRESH_SECONDS:.1f}s"),
    ]
    return "\n".join(lines)


def build_sweep_spec_text(spec: SweepSpec) -> str:
    return "\n".join([
        _fmt_field("r count", len(spec.r_values)),
        _fmt_field("r min/max", f"{display_float(spec.r_values[0], 3)} → {display_float(spec.r_values[-1], 3)}"),
        _fmt_field("α range", f"{display_float(spec.alpha_values[0], 3)} → {display_float(spec.alpha_values[-1], 3)}"),
        _fmt_field("α count", len(spec.alpha_values)),
        _fmt_field("seeds / cell", spec.seeds_per_cell),
        _fmt_field("intended total", spec.expected_total),
    ])


def build_latest_row_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data yet."

    latest = df.iloc[-1]
    preferred = [
        ("r", "r"),
        ("alpha", "α"),
        ("seed", "seed"),
        ("piF_tail", "πF_tail"),
        ("H_joint_mean", "H_joint"),
        ("best_corr", "best_corr"),
        ("corr0", "corr0"),
        ("delta_r2_freeze", "ΔR²_freeze"),
        ("delta_r2_entropy", "ΔR²_entropy"),
        ("K_max", "K_max"),
    ]

    lines: list[str] = []
    for col, label in preferred:
        if col not in df.columns:
            continue
        value = latest[col]
        if pd.isna(value):
            continue

        if col in {"r", "alpha"}:
            shown = display_float(float(value), 6)
        elif isinstance(value, float):
            shown = f"{float(value):.6g}"
        else:
            shown = str(value)

        lines.append(_fmt_field(label, shown))
    return "\n".join(lines)


def _col(text: str) -> str:
    return f"{text:^{COLUMN_WIDTH}}"


def _coverage_cell(count: int, total: int, selected: bool = False) -> str:
    frac = 0.0 if total <= 0 else count / total

    if frac <= 0.0:
        inner = "·"
    elif frac < 0.25:
        inner = "░"
    elif frac < 0.50:
        inner = "▒"
    elif frac < 0.75:
        inner = "▓"
    elif frac < 1.0:
        inner = "█"
    else:
        inner = "█"

    return _col(f"{inner}") if selected else _col(f" {inner} ")


def build_coverage_text(df: pd.DataFrame, spec: SweepSpec, selection: DetailSelection) -> str:
    if df.empty:
        return "No rows loaded."

    if not {"r", "alpha"}.issubset(df.columns):
        return "index.csv missing required columns: r, alpha"

    work = _safe_numeric(df, ["r", "alpha", "seed"]).dropna(subset=["r", "alpha"])

    if "seed" in work.columns:
        grouped = work.groupby(["r", "alpha"])["seed"].nunique().reset_index(name="n")
    else:
        grouped = work.groupby(["r", "alpha"]).size().reset_index(name="n")

    lookup = {
        (round(float(row.r), 12), round(float(row.alpha), 12)): int(row.n)
        for row in grouped.itertuples(index=False)
    }

    row_label_width = 8

    header_cells = []
    for a in spec.alpha_values:
        label = display_float(a, 3)
        if selection.selected_alpha is not None and abs(a - selection.selected_alpha) < 1e-12:
            label = f"[ {label} ]"
        header_cells.append(_col(label))

    header = "r \\ α".ljust(row_label_width) + " " + " ".join(header_cells)
    sep = "-" * len(header)

    lines = [header, sep]

    for r in spec.r_values:
        marker = "▶" if selection.selected_r is not None and abs(r - selection.selected_r) < 1e-12 else ">"
        row_prefix = f"{marker} {display_float(r, 3):>4}".ljust(row_label_width)

        cells = []
        for a in spec.alpha_values:
            n = lookup.get((round(r, 12), round(a, 12)), 0)
            selected = (
                selection.mode in {"cell", "trajectory"}
                and selection.selected_r is not None
                and selection.selected_alpha is not None
                and abs(r - selection.selected_r) < 1e-12
                and abs(a - selection.selected_alpha) < 1e-12
            )
            cells.append(_coverage_cell(n, spec.seeds_per_cell, selected=selected))

        lines.append(row_prefix + " " + " ".join(cells))

    lines.append("")
    lines.append(
        "Legend: "
        "· empty   "
        "░ <25%   "
        "▒ <50%   "
        "▓ <75%   "
        "█ <100%   "
        "█ full"
    )
    lines.append(f"Grid: {len(spec.r_values)} × {len(spec.alpha_values)} × {spec.seeds_per_cell}")
    return "\n".join(lines)


class PAMTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #left {
        width: 42;
        min-width: 36;
    }

    #right {
        width: 1fr;
    }

    Panel {
        border: round $accent;
        padding: 1 2;
        margin: 0 1 1 1;
    }

    #coverage {
        height: auto;
        min-height: 13;
    }

    #detail {
        height: 1fr;
    }

    #status, #spec, #latest {
        height: auto;
    }
    """

    BINDINGS = [
        ("up", "prev_r", "Prev r"),
        ("down", "next_r", "Next r"),
        ("left", "prev_alpha", "Prev α"),
        ("right", "next_alpha", "Next α"),
        ("enter", "toggle_mode", "Toggle mode"),
        ("t", "trajectory_mode", "Trajectory"),
    ]

    refresh_started_at = reactive(0.0)

    def __init__(self):
        super().__init__()
        self.snap = load_snapshot(INDEX_PATH)
        spec = self.snap.sweep_spec
        self.selection = DetailSelection(
            mode="row",
            selected_r=spec.r_values[0] if spec.r_values else None,
            selected_alpha=spec.alpha_values[0] if spec.alpha_values else None,
            selected_seed=0,
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            with Vertical(id="left"):
                self.status_panel = Panel("Run status", id="status")
                self.spec_panel = Panel("Sweep spec", id="spec")
                self.latest_panel = Panel("Latest row", id="latest")
                yield self.status_panel
                yield self.spec_panel
                yield self.latest_panel

            with Vertical(id="right"):
                self.coverage_panel = Panel("Seed coverage", id="coverage")
                self.detail_view = DetailView()
                yield self.coverage_panel
                yield self.detail_view

        yield Footer()

    def on_mount(self) -> None:
        self.title = "PAM Observatory"
        self.sub_title = "Live batch monitor"
        self.refresh_started_at = time()
        self.refresh_data()
        self.set_interval(REFRESH_SECONDS, self.refresh_data)

    def _ensure_selection_valid(self) -> None:
        spec = self.snap.sweep_spec
        if spec.r_values and self.selection.selected_r not in spec.r_values:
            self.selection.selected_r = spec.r_values[0]
        if spec.alpha_values and self.selection.selected_alpha not in spec.alpha_values:
            self.selection.selected_alpha = spec.alpha_values[0]

    def action_prev_r(self) -> None:
        values = self.snap.sweep_spec.r_values
        if not values:
            return
        cur = self.selection.selected_r if self.selection.selected_r is not None else values[0]
        idx = values.index(cur)
        self.selection.selected_r = values[max(0, idx - 1)]
        self.refresh_data()

    def action_next_r(self) -> None:
        values = self.snap.sweep_spec.r_values
        if not values:
            return
        cur = self.selection.selected_r if self.selection.selected_r is not None else values[0]
        idx = values.index(cur)
        self.selection.selected_r = values[min(len(values) - 1, idx + 1)]
        self.refresh_data()

    def action_prev_alpha(self) -> None:
        values = self.snap.sweep_spec.alpha_values
        if not values:
            return
        cur = self.selection.selected_alpha if self.selection.selected_alpha is not None else values[0]
        idx = values.index(cur)
        self.selection.selected_alpha = values[max(0, idx - 1)]
        self.refresh_data()

    def action_next_alpha(self) -> None:
        values = self.snap.sweep_spec.alpha_values
        if not values:
            return
        cur = self.selection.selected_alpha if self.selection.selected_alpha is not None else values[0]
        idx = values.index(cur)
        self.selection.selected_alpha = values[min(len(values) - 1, idx + 1)]
        self.refresh_data()

    def action_toggle_mode(self) -> None:
        if self.selection.mode == "row":
            self.selection.mode = "cell"
        elif self.selection.mode == "cell":
            self.selection.mode = "row"
        elif self.selection.mode == "trajectory":
            self.selection.mode = "cell"
        self.refresh_data()

    def action_trajectory_mode(self) -> None:
        self.selection.mode = "trajectory"
        self.refresh_data()

    def refresh_data(self) -> None:
        self.snap = load_snapshot(INDEX_PATH)
        self._ensure_selection_valid()

        elapsed = time() - self.refresh_started_at
        qph = self.snap.completed / elapsed * 3600.0 if elapsed > 0 else 0.0

        self.status_panel.set_body(build_status_text(self.snap, self.selection, qph))
        self.spec_panel.set_body(build_sweep_spec_text(self.snap.sweep_spec))
        self.latest_panel.set_body(build_latest_row_text(self.snap.df))
        self.coverage_panel.set_body(build_coverage_text(self.snap.df, self.snap.sweep_spec, self.selection))
        self.detail_view.update_detail(self.snap.df, self.selection, alpha_values=self.snap.sweep_spec.alpha_values)


if __name__ == "__main__":
    PAMTUI().run()
