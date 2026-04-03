from __future__ import annotations

import pandas as pd
from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.views.formatting import fmt_value, overlay_label


class RankingView(Static):
    def render_from_overlay(self, overlay_name: str, ranking_df: pd.DataFrame) -> None:
        table = Table(expand=True)
        table.add_column("#", justify="right", width=3)
        if "node_id" in ranking_df.columns:
            table.add_column("node", justify="right", width=6)
        table.add_column("r", justify="right", width=7)
        table.add_column("α", justify="right", width=10)
        table.add_column("value", justify="right", width=10)

        if ranking_df.empty:
            table.add_row("-", "-", "-", "-") if "node_id" in ranking_df.columns else table.add_row("-", "-", "-")
        else:
            for _, row in ranking_df.iterrows():
                cells = [str(int(row["rank"]))]
                if "node_id" in ranking_df.columns:
                    cells.append(str(row["node_id"]))
                cells.extend([
                    fmt_value(row["r"], 3),
                    fmt_value(row["alpha"], 6),
                    fmt_value(row["value"], 3),
                ])
                table.add_row(*cells)

        self.update(
            Panel(
                table,
                title=f"Ranking — {overlay_label(overlay_name)}",
                border_style="magenta",
            )
        )
