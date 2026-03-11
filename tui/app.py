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

# Adjust these if your sweep changes.
R_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
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
    coverage_table_text: str


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_coverage_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows loaded."

    if not {"r", "alpha"}.issubset(df.columns):
        return "index.csv is missing required columns: r, alpha"

    work = _safe_numeric(df, ["r", "alpha", "seed"])

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
        (round(float(row.r), 6), round(float(row.alpha), 6)): int(row.n)
        for row in grouped.itertuples(index=False)
    }

    cell_width = 5
    header = "r \\ α".ljust(7) + " ".join(f"{a:>{cell_width}.2f}" for a in ALPHA_VALUES)
    sep = "-" * len(header)

    lines = [header, sep]
    for r in R_VALUES:
        row = [f"{r:>5.2f}  "]
        for a in ALPHA_VALUES:
            n = lookup.get((round(r, 6), round(a, 6)), 0)
            row.append(f"{n:>{cell_width}d}")
        lines.append(" ".join(row))

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
        if isinstance(value, float):
            lines.append(f"{label:<12} {value:.6g}")
        else:
            lines.append(f"{label:<12} {value}")

    return "\n".join(lines) if lines else "No known metric columns found."


def load_snapshot(index_path: Path) -> Snapshot:
    expected_total = len(R_VALUES) * len(ALPHA_VALUES) * EXPECTED_SEEDS_PER_CELL

    if not index_path.exists():
        return Snapshot(
            row_count=0,
            completed=0,
            expected_total=expected_total,
            percent=0.0,
            last_modified="missing",
            latest_metrics_text="index.csv not found.",
            coverage_table_text="Waiting for outputs/index.csv ...",
        )

    try:
        df = pd.read_csv(index_path)
    except Exception as exc:
        return Snapshot(
            row_count=0,
            completed=0,
            expected_total=expected_total,
            percent=0.0,
            last_modified="unreadable",
            latest_metrics_text=f"Failed to read CSV:\n{exc}",
            coverage_table_text="No coverage available.",
        )

    completed = len(df)
    percent = 100.0 * completed / expected_total if expected_total else 0.0
    mtime = pd.Timestamp(index_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")

    return Snapshot(
        row_count=len(df),
        completed=completed,
        expected_total=expected_total,
        percent=percent,
        last_modified=mtime,
        latest_metrics_text=build_latest_metrics_text(df),
        coverage_table_text=build_coverage_table(df),
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
            f"throughput      {qph:6.2f} quenches/hr\n"
            f"last modified   {snap.last_modified}\n"
            f"refresh every   {REFRESH_SECONDS:.1f}s"
        )

        self.status_panel.set_body(status_text)
        self.latest_panel.set_body(snap.latest_metrics_text)
        self.coverage_panel.set_body(f"[code]{snap.coverage_table_text}[/code]")


if __name__ == "__main__":
    PAMTUI().run()
