from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from tui.ascii_plot import render_ascii_plot
from tui.models import SweepSpec


def display_float(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.rstrip("0").rstrip(".")


def sparkbar(value: float, vmin: float, vmax: float, width: int = 6) -> str:
    if vmax <= vmin:
        return "█" + " " * (width - 1)
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    filled = max(1, int(round(t * width)))
    filled = min(width, filled)
    return "█" * filled + " " * (width - filled)


class DetailView(Static):
    def __init__(self, spec: SweepSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.mode = "empty"
        self.payload: dict = {}

    def show_empty(self) -> None:
        self.mode = "empty"
        self.payload = {}
        self.update(self.render_detail())

    def show_row_detail(self, row_summary: dict) -> None:
        self.mode = "row"
        self.payload = row_summary
        self.update(self.render_detail())

    def show_cell_detail(self, cell_summary: dict) -> None:
        self.mode = "cell"
        self.payload = cell_summary
        self.update(self.render_detail())

    def show_trajectory_detail(self, traj_summary: dict) -> None:
        self.mode = "trajectory"
        self.payload = traj_summary
        self.update(self.render_detail())

    def _build_metric_lines(self, series: list[tuple[float, float | None]], label: str) -> list[str]:
        lines = [label]
        values = [v for _, v in series if v is not None]
        if not values:
            lines.append("  no data")
            return lines
        vmin = min(values)
        vmax = max(values)
        for alpha, value in series:
            if value is None:
                lines.append(f"{display_float(alpha, 3):>6}  ·")
            else:
                bar = sparkbar(float(value), float(vmin), float(vmax), width=6)
                lines.append(f"{display_float(alpha, 3):>6}  {bar} {value:.4f}")
        return lines

    def _merge_four_columns(self, cols, width=24, gap="  "):
        max_rows = max(len(c) for c in cols)
        out = []
        for i in range(max_rows):
            row = []
            for c in cols:
                val = c[i] if i < len(c) else ""
                row.append(f"{val:<{width}}")
            out.append(gap.join(row))
        return out

    def render_empty(self) -> Text:
        text = Text()
        text.append("Detail view", style="bold")
        text.append("\n\nNo selection yet.", style="dim")
        return text

    def render_row_detail(self) -> Text:
        text = Text()
        row_summary = self.payload

        r_value = row_summary.get("r")
        metrics = row_summary.get("metrics", {})
        coverage = row_summary.get("coverage", {})
        alpha_values = row_summary.get("alpha_values", self.spec.alpha_values)

        text.append(f"Detail: row mode   r = {display_float(r_value, 3)}", style="bold")
        text.append("\n\n")
        completed_cells = coverage.get("completed_cells", 0)
        total_cells = coverage.get("total_cells", len(alpha_values))
        min_seed_count = coverage.get("min_seed_count", 0)
        max_seed_count = coverage.get("max_seed_count", 0)

        text.append(
            f"α cells observed: {completed_cells} / {total_cells}    seed coverage: min {min_seed_count}   max {max_seed_count}",
            style="cyan",
        )
        text.append("\n\n")

        columns = [
            self._build_metric_lines(metrics.get("piF_tail_mean", []), "πF_tail"),
            self._build_metric_lines(metrics.get("H_joint_mean_mean", []), "H_joint"),
            self._build_metric_lines(metrics.get("best_corr_mean", []), "best_corr"),
            self._build_metric_lines(metrics.get("delta_r2_freeze_mean", []), "ΔR²_freeze"),
        ]

        for line in self._merge_four_columns(columns):
            text.append(line)
            text.append("\n")
        return text

    def render_cell_detail(self) -> Text:
        text = Text()
        cell = self.payload

        r_value = cell.get("r")
        alpha_value = cell.get("alpha")
        seed_count = cell.get("seed_count", 0)
        seed_target = cell.get("seed_target", self.spec.seeds_per_cell)
        metrics = cell.get("metrics", {})
        per_seed = cell.get("per_seed", [])

        text.append(
            f"Detail: cell mode   r = {display_float(r_value, 3)}   α = {display_float(alpha_value, 3)}",
            style="bold",
        )
        text.append("\n\n")
        text.append(f"seed coverage: {seed_count} / {seed_target}", style="cyan")
        text.append("\n\n")
        text.append("cell means\n", style="bold")

        metric_order = [
            ("piF_tail", "πF_tail"),
            ("H_joint_mean", "H_joint"),
            ("best_corr", "best_corr"),
            ("corr0", "corr0"),
            ("delta_r2_freeze", "ΔR²_freeze"),
            ("delta_r2_entropy", "ΔR²_entropy"),
            ("K_max", "K_max"),
        ]
        for key, label in metric_order:
            value = metrics.get(key)
            if value is not None:
                text.append(f"{label:<14} {value:.6g}\n")

        if per_seed:
            text.append("\nper-seed\n", style="bold")
            text.append("seed   πF_tail   H_joint   best_corr   ΔR²_freeze\n", style="dim")
            for row in per_seed[:12]:
                seed = row.get("seed", "")
                pif = row.get("piF_tail")
                hj = row.get("H_joint_mean")
                bc = row.get("best_corr")
                dr2 = row.get("delta_r2_freeze")
                text.append(
                    f"{str(seed):<6} "
                    f"{'' if pif is None else f'{pif:.4f}':<9} "
                    f"{'' if hj is None else f'{hj:.4f}':<9} "
                    f"{'' if bc is None else f'{bc:.4f}':<11} "
                    f"{'' if dr2 is None else f'{dr2:.4f}':<9}\n"
                )

        return text

    def render_trajectory_detail(self) -> Text:
        text = Text()
        traj = self.payload
        r_value = traj.get("r")
        alpha_value = traj.get("alpha")
        seed = traj.get("seed")
        series = traj.get("series", {})

        text.append(
            f"Detail: trajectory mode   r = {display_float(r_value, 3)}   α = {display_float(alpha_value, 3)}   seed = {seed}",
            style="bold",
        )
        text.append("\n\n")

        plots = [
            ("F_raw", "F_raw(t)"),
            ("H_joint", "H_joint(t)"),
            ("K", "K(t)"),
            ("pi", "π_F smooth(t)"),
        ]

        has_any = False
        for key, title in plots:
            vals = series.get(key)
            if vals is None:
                continue
            has_any = True
            for line in render_ascii_plot(vals, title=title, width=64, height=8):
                text.append(line)
                text.append("\n")
            text.append("\n")

        if not has_any:
            text.append("No trajectory file available for this cell.", style="dim")

        return text

    def render_detail(self) -> Text:
        if self.mode == "row":
            return self.render_row_detail()
        if self.mode == "cell":
            return self.render_cell_detail()
        if self.mode == "trajectory":
            return self.render_trajectory_detail()
        return self.render_empty()

    def on_mount(self) -> None:
        self.update(self.render_detail())
