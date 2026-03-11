from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable

import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static


INDEX_PATH = Path("outputs/index.csv")
REFRESH_SECONDS = 5.0

# Declared default sweep spec for the currently running experiment.
DEFAULT_R_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_ALPHA_VALUES = [
    0.03,
    0.03857142857142857,
    0.04714285714285714,
    0.055714285714285716,
    0.06428571428571428,
    0.07285714285714286,
    0.08142857142857143,
    0.09,
    0.09857142857142857,
    0.10714285714285714,
    0.11571428571428571,
    0.12428571428571428,
    0.13285714285714287,
    0.14142857142857143,
    0.15,
]
DEFAULT_SEEDS_PER_CELL = 10


@dataclass(frozen=True)
class SweepSpec:
    r_values: list[float]
    alpha_values: list[float]
    seeds_per_cell: int

    @property
    def expected_total(self) -> int:
        return len(self.r_values) * len(self.alpha_values) * self.seeds_per_cell


DEFAULT_SWEEP = SweepSpec(
    r_values=DEFAULT_R_VALUES,
    alpha_values=DEFAULT_ALPHA_VALUES,
    seeds_per_cell=DEFAULT_SEEDS_PER_CELL,
)


@dataclass
class Snapshot:
    row_count: int
    completed: int
    expected_total: int
    percent: float
    last_modified: str
    latest_metrics_text: str
    coverage_heatmap_text: str
    sweep_spec_text: str
    observed_grid_text: str


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
    s = f"{x:.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    return s


def coverage_cell(count: int, total: int) -> str:
    frac = 0.0 if total <= 0 else count / total

    if frac <= 0.0:
        return "[dim]·[/dim]"
    if frac < 0.25:
        return "[cyan]░[/cyan]"
    if frac < 0.50:
        return "[green]▒[/green]"
    if frac < 0.75:
        return "[yellow]▓[/yellow]"
    if frac < 1.0:
        return "[magenta]█[/magenta]"
    return "[bold red]█[/bold red]"


def build_sweep_spec_text(spec: SweepSpec) -> str:
    r_text = ", ".join(display_float(r, 3) for r in spec.r_values)
    alpha_min = display_float(min(spec.alpha_values), 3)
    alpha_max = display_float(max(spec.alpha_values), 3)

    return (
        f"r values        {r_text}\n"
        f"α range         {alpha_min} → {alpha_max}\n"
        f"α count         {len(spec.alpha_values)}\n"
        f"seeds / cell    {spec.seeds_per_cell}\n"
        f"intended total  {spec.expected_total}"
    )


def build_observed_grid_text(df: pd.DataFrame, spec: SweepSpec) -> str:
    if df.empty or not {"r", "alpha"}.issubset(df.columns):
        return (
            f"observed r      0 / {len(spec.r_values)}\n"
            f"observed α      0 / {len(spec.alpha_values)}"
        )

    work = _safe_numeric(df, ["r", "alpha"]).dropna(subset=["r", "alpha"])
    observed_r = _sorted_unique_numeric(work["r"])
    observed_alpha = _sorted_unique_numeric(work["alpha"])

    return (
        f"observed r      {len(observed_r)} / {len(spec.r_values)}\n"
        f"observed α      {len(observed_alpha)} / {len(spec.alpha_values)}"
    )


def build_coverage_heatmap(df: pd.DataFrame, spec: SweepSpec) -> str:
    if df.empty:
        alpha_labels = [display_float(a, 3) for a in spec.alpha_values]
        row_label_width = 8
        cell_width = 4

        header = "r \\ α".ljust(row_label_width) + " " + " ".join(
            f"{label:>{cell_width}}" for label in alpha_labels
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for r in spec.r_values:
            row = [f"{display_float(r, 3):>{row_label_width-1}} "]
            for _a in spec.alpha_values:
                row.append(f"{coverage_cell(0, spec.seeds_per_cell):^{cell_width}}")
            lines.append(" ".join(row))

        lines.append("")
        lines.append(
            "[dim]Legend:[/dim] "
            "[dim]·[/dim] empty   "
            "[cyan]░[/cyan] <25%   "
            "[green]▒[/green] <50%   "
            "[yellow]▓[/yellow] <75%   "
            "[magenta]█[/magenta] <100%   "
            "[bold red]█[/bold red] complete"
        )
        lines.append(
            f"[dim]Grid:[/dim] {len(spec.r_values)} r-values × "
            f"{len(spec.alpha_values)} α-values × {spec.seeds_per_cell} seeds"
        )
        return "\n".join(lines)

    if not {"r", "alpha"}.issubset(df.columns):
        return "index.csv is missing required columns: r, alpha"

    work = _safe_numeric(df, ["r", "alpha", "seed"])
    work = work.dropna(subset=["r", "alpha"])

    if "seed" in work.columns:
        grouped = (
            work.groupby(["r", "alpha"])["seed"]
            .nunique()
            .reset_index(name="n")
        )
    else:
        grouped = (
            work.groupby(["r", "alpha"])
            .size()
            .reset_index(name="n")
        )

    lookup = {
        (round(float(row.r), 12), round(float(row.alpha), 12)): int(row.n)
        for row in grouped.itertuples(index=False)
    }

    alpha_labels = [display_float(a, 3) for a in spec.alpha_values]
    row_label_width = 8
    cell_width = 4

    header = "r \\ α".ljust(row_label_width) + " " + " ".join(
        f"{label:>{cell_width}}" for label in alpha_labels
    )
    sep = "-" * len(header)

    lines = [header, sep]

    for r in spec.r_values:
        row = [f"{display_float(r, 3):>{row_label_width-1}} "]
        for a in spec.alpha_values:
            n = lookup.get((round(r, 12), round(a, 12)), 0)
            row.append(f"{coverage_cell(n, spec.seeds_per_cell):^{cell_width}}")
        lines.append(" ".join(row))

    lines.append("")
    lines.append(
        "[dim]Legend:[/dim] "
        "[dim]·[/dim] empty   "
        "[cyan]░[/cyan] <25%   "
        "[green]▒[/green] <50%   "
        "[yellow]▓[/yellow] <75%   "
        "[magenta]█[/magenta] <100%   "
        "[bold red]█[/bold red] complete"
    )
    lines.append(
        f"[dim]Grid:[/dim] {len(spec.r_values)} r-values × "
        f"{len(spec.alpha_values)} α-values × {spec.seeds_per_cell} seeds"
    )

    return "\n".join(lines)


def build_latest_metrics_text(df: pd.DataFrame) -> str:
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

    lines = []
    for col, label in preferred:
        if col not in df.columns:
            continue
        value = latest[col]
        if pd.isna(value):
            continue

        if col in {"r", "alpha"}:
            lines.append(f"{label:<12} {display_float(float(value), 6)}")
        elif isinstance(value, float):
            lines.append(f"{label:<12} {value:.6g}")
        else:
            lines.append(f"{label:<12} {value}")

    return "\n".join(lines) if lines else "No known metric columns found."


def load_snapshot(index_path: Path, spec: SweepSpec) -> Snapshot:
    if not index_path.exists():
        return Snapshot(
            row_count=0,
            completed=0,
            expected_total=spec.expected_total,
            percent=0.0,
            last_modified="missing",
            latest_metrics_text="index.csv not found.",
            coverage_heatmap_text=build_coverage_heatmap(pd.DataFrame(), spec),
            sweep_spec_text=build_sweep_spec_text(spec),
            observed_grid_text=(
                f"observed r      0 / {len(spec.r_values)}\n"
                f"observed α      0 / {len(spec.alpha_values)}"
            ),
        )

    try:
        df = pd.read_csv(index_path)
    except Exception as exc:
        return Snapshot(
            row_count=0,
            completed=0,
            expected_total=spec.expected_total,
            percent=0.0,
            last_modified="unreadable",
            latest_metrics_text=f"Failed to read CSV:\n{exc}",
            coverage_heatmap_text="No coverage available.",
            sweep_spec_text=build_sweep_spec_text(spec),
            observed_grid_text=(
                f"observed r      0 / {len(spec.r_values)}\n"
                f"observed α      0 / {len(spec.alpha_values)}"
            ),
        )

    completed = len(df)
    percent = 100.0 * completed / spec.expected_total if spec.expected_total else 0.0
    mtime = pd.Timestamp(index_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")

    return Snapshot(
        row_count=len(df),
        completed=completed,
        expected_total=spec.expected_total,
        percent=percent,
        last_modified=mtime,
        latest_metrics_text=build_latest_metrics_text(df),
        coverage_heatmap_text=build_coverage_heatmap(df, spec),
        sweep_spec_text=build_sweep_spec_text(spec),
        observed_grid_text=build_observed_grid_text(df, spec),
    )


class Panel(Static):
    def __init__(self, title: str, body: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.body = body

    def set_body(self, body: str) -> None:
        self.body = body
        self.update(self.render_text())

    def render_text(self) -> str:
        return f"[b]{self.title}[/b]\n\n{self.body}"

    def on_mount(self) -> None:
        self.update(self.render_text())


class PAMTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #left {
        width: 36;
        min-width: 34;
    }

    #right {
        width: 1fr;
    }

    Panel {
        border: round $primary;
        padding: 1 2;
        margin: 0 1 1 1;
    }

    #coverage {
        height: 1fr;
    }

    #latest {
        height: auto;
    }

    #status {
        height: auto;
    }

    #spec {
        height: auto;
    }
    """

    refresh_started_at = reactive(0.0)

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

            with Container(id="right"):
                self.coverage_panel = Panel("Seed coverage", id="coverage")
                yield self.coverage_panel

        yield Footer()

    def on_mount(self) -> None:
        self.title = "PAM Observatory"
        self.sub_title = "Live batch monitor"
        self.refresh_started_at = time()
        self.refresh_data()
        self.set_interval(REFRESH_SECONDS, self.refresh_data)

    def refresh_data(self) -> None:
        snap = load_snapshot(INDEX_PATH, DEFAULT_SWEEP)
        elapsed = time() - self.refresh_started_at
        qph = snap.completed / elapsed * 3600.0 if elapsed > 0 else 0.0

        status_text = (
            f"index path     {INDEX_PATH}\n"
            f"rows loaded     {snap.row_count}\n"
            f"completed       {snap.completed} / {snap.expected_total}\n"
            f"progress        {snap.percent:6.2f}%\n"
            f"throughput      {qph:8.2f}\n"
            f"quenches/hr\n"
            f"{snap.observed_grid_text}\n"
            f"last modified   {snap.last_modified}\n"
            f"refresh every   {REFRESH_SECONDS:.1f}s"
        )

        self.status_panel.set_body(status_text)
        self.spec_panel.set_body(snap.sweep_spec_text)
        self.latest_panel.set_body(snap.latest_metrics_text)
        self.coverage_panel.set_body(snap.coverage_heatmap_text)


if __name__ == "__main__":
    PAMTUI().run()