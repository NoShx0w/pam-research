from __future__ import annotations

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.state import ObservatoryState


class ManifoldView(Static):
    def _render_scalar_grid(self, state: ObservatoryState, df, value_col: str) -> Table:
        table = Table.grid(padding=(0, 1))
        for _ in range(state.grid_cols):
            table.add_column(justify="center", width=2)

        lookup = {}
        if df is not None and not df.empty and {"r", "alpha", value_col}.issubset(df.columns):
            r_vals = sorted(df["r"].dropna().unique())
            a_vals = sorted(df["alpha"].dropna().unique())
            for i, r in enumerate(r_vals):
                for j, a in enumerate(a_vals):
                    hit = df[(df["r"] == r) & (df["alpha"] == a)]
                    if hit.empty:
                        lookup[(i, j)] = None
                    else:
                        lookup[(i, j)] = hit.iloc[0][value_col]

        for i in range(state.grid_rows):
            row = []
            for j in range(state.grid_cols):
                val = lookup.get((i, j), None)

                if i == state.selected_i and j == state.selected_j:
                    row.append("[black on bright_white]●[/]")
                elif val is None or val != val:
                    row.append("[dim]·[/]")
                else:
                    row.append("[green]■[/]")

            table.add_row(*row)
        return table

    def _render_run_grid(self, state: ObservatoryState, run_data) -> Table:
        table = Table.grid(padding=(0, 1))
        for _ in range(state.grid_cols):
            table.add_column(justify="center", width=2)

        coverage_lookup = {}
        if run_data is not None and not run_data.coverage_df.empty:
            r_vals = sorted(run_data.coverage_df["r"].dropna().unique())
            a_vals = sorted(run_data.coverage_df["alpha"].dropna().unique())
            for i, r in enumerate(r_vals):
                for j, a in enumerate(a_vals):
                    hit = run_data.coverage_df[
                        (run_data.coverage_df["r"] == r) & (run_data.coverage_df["alpha"] == a)
                    ]
                    coverage_lookup[(i, j)] = 0 if hit.empty else int(hit.iloc[0]["n_rows"])

        for i in range(state.grid_rows):
            row = []
            for j in range(state.grid_cols):
                count = coverage_lookup.get((i, j), 0)
                if i == state.selected_i and j == state.selected_j:
                    row.append("[black on bright_white]●[/]")
                elif count > 0:
                    row.append("[green]■[/]")
                else:
                    row.append("[dim]·[/]")
            table.add_row(*row)
        return table

    def _render_mds_placeholder(self, state: ObservatoryState) -> Table:
        table = Table.grid(padding=(0, 1))
        cols = max(12, state.grid_cols + 4)
        rows = max(8, state.grid_rows)
        for _ in range(cols):
            table.add_column(justify="center", width=2)

        points = {
            (1, 2), (2, 4), (2, 9), (3, 6), (3, 12),
            (4, 3), (4, 10), (5, 7), (6, 5), (6, 11),
            (7, 8), (7, 13),
        }
        sel = (min(rows - 1, state.selected_i % rows), min(cols - 1, state.selected_j % cols))

        for i in range(rows):
            row = []
            for j in range(cols):
                if (i, j) == sel:
                    row.append("[black on bright_white]●[/]")
                elif (i, j) in points:
                    row.append("[cyan]•[/]")
                else:
                    row.append(" ")
            table.add_row(*row)
        return table

    def render_from_state(self, state: ObservatoryState, mode_data=None) -> None:
        if state.view_space == "mds":
            content = self._render_mds_placeholder(state)
        elif state.mode == "Run":
            content = self._render_run_grid(state, mode_data)
        elif state.mode == "Geometry":
            value_col = "scalar_curvature" if state.overlay == "curvature" else "fim_det"
            content = self._render_scalar_grid(state, mode_data.geometry_df if mode_data else None, value_col)
        elif state.mode == "Phase":
            value_col = "signed_phase" if state.overlay == "signed_phase" else "distance_to_seam"
            content = self._render_scalar_grid(state, mode_data.phase_df if mode_data else None, value_col)
        else:
            content = self._render_run_grid(state, None)

        title = f"Manifold — {state.mode} / {state.view_space.upper()} / {state.overlay}"
        self.update(Panel(Align.center(content, vertical="top"), title=title, border_style="green"))