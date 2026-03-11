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
EXPECTED_SEEDS_PER_CELL = 10
REFRESH_SECONDS = 5.0


@dataclass
class Snapshot:
    row_count: int
    completed: int
    expected_total: int
    percent: float
    last_modified: str
    latest_metrics_text: str
    coverage_heatmap_text: str


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


def build_coverage_heatmap(df: pd.DataFrame) -> tuple[str, int]:
    if df.empty:
        return "No rows loaded.", 0

    if not {"r", "alpha"}.issubset(df.columns):
        return "index.csv is missing required columns: r, alpha", 0

    work = _safe_numeric(df, ["r", "alpha", "seed"])
    work = work.dropna(subset=["r", "alpha"])

    r_values = _sorted_unique_numeric(work["r"])
    alpha_values = _sorted_unique_numeric(work["alpha"])

    expected_total = len(r_values) * len(alpha_values) * EXPECTED_SEEDS_PER_CELL

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

    alpha_labels = [display_float(a, 3) for a in alpha_values]
    row_label_width = 8
    cell_width = 4

    header = "r \\ α".ljust(row_label_width) + " " + " ".join(
        f"{label:>{cell_width}}" for label in alpha_labels
    )
    sep = "-" * len(header)

    lines = [header, sep]

    for r in r_values:
        row = [f"{display_float(r, 3):>{row_label_width-1}} "]
        for a in alpha_values:
            n = lookup.get((round(r, 12), round(a, 12)), 0)
            row.append(f"{coverage_cell(n, EXPECTED_SEEDS_PER_CELL):^{cell_width}}")
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
        f"[dim]Grid:[/dim] {len(r_values)} r-values × {len(alpha_values)} α-values × {EXPECTED_SEEDS_PER_CELL} seeds"
    )

    return "\n".join(lines), expected_total


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


def load_snapshot(index_path: Path) -> Snapshot:
    if not index_path.exists():
        return Snapshot(
            row_count=0,
            completed=0,
            expected_total=0,
            percent=0.0,
            last_modified="missing",
            latest_metrics_text="index.csv not found.",
            coverage_heatmap_text="Waiting for outputs/index.csv ...",
        )

    try:
        df = pd.read_csv(index_path)
    except Exception as exc:
        return Snapshot(
            row_count=0,
            completed=0,
            expected_total=0,
            percent=0.0,
            last_modified="unreadable",
            latest_metrics_text=f"Failed to read CSV:\n{exc}",
            coverage_heatmap_text="No coverage available.",
        )

    completed = len(df)
    coverage_heatmap_text, expected_total = build_coverage_heatmap(df)
    percent = 100.0 * completed / expected_total if expected_total else 0.0
    mtime = pd.Timestamp(index_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")

    return Snapshot(
        row_count=len(df),
        completed=completed,
        expected_total=expected_total,
        percent=percent,
        last_modified=mtime,
        latest_metrics_text=build_latest_metrics_text(df),
        coverage_heatmap_text=coverage_heatmap_text,
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
        width: 34;
        min-width: 30;
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
    """

    refresh_started_at = reactive(0.0)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            with Vertical(id="left"):
                self.status_panel = Panel("Run status", id="status")
                self.latest_panel = Panel("Latest row", id="latest")
                yield self.status_panel
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
        snap = load_snapshot(INDEX_PATH)
        elapsed = time() - self.refresh_started_at
        qph = snap.completed / elapsed * 3600.0 if elapsed > 0 else 0.0

        status_text = (
            f"index path     {INDEX_PATH}\n"
            f"rows loaded     {snap.row_count}\n"
            f"completed       {snap.completed} / {snap.expected_total}\n"
            f"progress        {snap.percent:6.2f}%\n"
            f"throughput      {qph:8.2f}\n"
            f"quenches/hr\n"
            f"last modified   {snap.last_modified}\n"
            f"refresh every   {REFRESH_SECONDS:.1f}s"
        )

        self.status_panel.set_body(status_text)
        self.latest_panel.set_body(snap.latest_metrics_text)
        self.coverage_panel.set_body(snap.coverage_heatmap_text)


if __name__ == "__main__":
    PAMTUI().run()