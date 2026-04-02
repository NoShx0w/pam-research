from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

from observatory.state import ObservatoryState
from observatory.views.scalars import render_signed_cell, render_unsigned_cell


class ManifoldView(Static):
    SIGNED_OVERLAYS = {
        "signed_phase",
    }

    def _geometry_value_col(self, overlay: str) -> str:
        if overlay == "curvature":
            return "scalar_curvature"
        if overlay == "condition_number":
            return "fim_cond"
        return "fim_det"

    def _phase_value_col(self, overlay: str) -> str:
        if overlay == "distance_to_seam":
            return "distance_to_seam"
        return "signed_phase"

    def _topology_value_col(self, overlay: str) -> str:
        return "criticality"

    def _operators_value_col(self, overlay: str) -> str:
        return "lazarus_score"

    def _identity_value_col(self, overlay: str) -> str:
        if overlay == "identity_magnitude":
            return "identity_magnitude"
        if overlay == "absolute_holonomy":
            return "absolute_holonomy_node"
        if overlay == "unsigned_local_obstruction":
            return "obstruction_mean_abs_holonomy"
        if overlay == "signed_local_obstruction":
            return "obstruction_signed_sum_holonomy"
        return "identity_spin"

    def _drawable_size(self) -> tuple[int, int]:
        width = max(10, self.size.width - 4)
        height = max(6, self.size.height - 2)
        return width, height

    def _lookup_from_df(self, df, value_col: str):
        lookup = {}
        r_vals = []
        a_vals = []

        if df is not None and not df.empty and {"r", "alpha", value_col}.issubset(df.columns):
            r_vals = sorted(df["r"].dropna().unique())
            a_vals = sorted(df["alpha"].dropna().unique())

            for i, r in enumerate(r_vals):
                for j, a in enumerate(a_vals):
                    hit = df[(df["r"] == r) & (df["alpha"] == a)]
                    lookup[(i, j)] = None if hit.empty else hit.iloc[0][value_col]

        return lookup, r_vals, a_vals

    def _render_grid_blocks(
        self,
        state: ObservatoryState,
        lookup: dict[tuple[int, int], object],
        *,
        signed: bool,
    ) -> Text:
        width, height = self._drawable_size()

        block_w = max(1, min(4, width // max(1, state.grid_cols)))
        block_h = max(1, min(3, height // max(1, state.grid_rows)))

        values = [v for v in lookup.values() if v is not None]
        if not values:
            values = [0.0]

        if signed:
            vabs = max(abs(float(v)) for v in values)
        else:
            vmin = min(float(v) for v in values)
            vmax = max(float(v) for v in values)

        out = Text()

        for i in range(state.grid_rows):
            for _ in range(block_h):
                line = Text()
                for j in range(state.grid_cols):
                    val = lookup.get((i, j), None)
                    selected = (i == state.selected_i and j == state.selected_j)

                    if signed:
                        cell = render_signed_cell(val, vabs=vabs, selected=selected)
                    else:
                        cell = render_unsigned_cell(val, vmin=vmin, vmax=vmax, selected=selected)

                    expanded = cell * block_w
                    line.append_text(Text.from_markup(expanded))

                out.append_text(line)
                out.append("\n")

        return out

    def _render_run_grid(self, state: ObservatoryState, run_data) -> Text:
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

        return self._render_grid_blocks(state, coverage_lookup, signed=False)

    def _render_scalar_grid(self, state: ObservatoryState, df, value_col: str, *, signed: bool) -> Text:
        lookup, _, _ = self._lookup_from_df(df, value_col)
        return self._render_grid_blocks(state, lookup, signed=signed)

    def _render_mds_real(self, state: ObservatoryState, mds_data) -> Text:
        if mds_data is None or mds_data.mds_df.empty or not {"mds1", "mds2"}.issubset(mds_data.mds_df.columns):
            return self._render_mds_placeholder(state)

        df = mds_data.mds_df.copy()
        df = df.dropna(subset=["mds1", "mds2", "r", "alpha"])
        if df.empty:
            return self._render_mds_placeholder(state)

        width, height = self._drawable_size()
        width = max(16, width)
        height = max(8, height)

        x_min, x_max = float(df["mds1"].min()), float(df["mds1"].max())
        y_min, y_max = float(df["mds2"].min()), float(df["mds2"].max())

        pad_x = 2
        pad_y = 1
        inner_w = max(4, width - 2 * pad_x)
        inner_h = max(4, height - 2 * pad_y)

        def scale_x(x: float) -> int:
            if x_max == x_min:
                return pad_x + inner_w // 2
            return pad_x + int(round((x - x_min) / (x_max - x_min) * (inner_w - 1)))

        def scale_y(y: float) -> int:
            if y_max == y_min:
                return pad_y + inner_h // 2
            return pad_y + int(round((y - y_min) / (y_max - y_min) * (inner_h - 1)))

        canvas = [[" " for _ in range(width)] for _ in range(height)]

        r_vals = sorted(df["r"].dropna().unique())
        a_vals = sorted(df["alpha"].dropna().unique())
        sel_r = r_vals[state.selected_i] if state.selected_i < len(r_vals) else None
        sel_a = a_vals[state.selected_j] if state.selected_j < len(a_vals) else None

        for _, row in df.iterrows():
            cx = scale_x(float(row["mds1"]))
            cy = scale_y(float(row["mds2"]))
            rr = height - 1 - cy

            if 0 <= rr < height and 0 <= cx < width:
                is_selected = (
                    sel_r is not None
                    and sel_a is not None
                    and row["r"] == sel_r
                    and row["alpha"] == sel_a
                )
                canvas[rr][cx] = "[black on bright_white]●[/]" if is_selected else "[cyan]•[/]"

        out = Text()
        for row in canvas:
            line = Text()
            for cell in row:
                line.append_text(Text.from_markup(cell))
            out.append_text(line)
            out.append("\n")
        return out

    def _render_mds_placeholder(self, state: ObservatoryState) -> Text:
        width, height = self._drawable_size()
        width = max(16, width)
        height = max(8, height)

        points = {
            (1, 2), (2, 4), (2, 9), (3, 6), (3, 12),
            (4, 3), (4, 10), (5, 7), (6, 5), (6, 11),
            (7, 8), (7, 13),
        }

        canvas = [[" " for _ in range(width)] for _ in range(height)]
        sel_r = min(height - 1, state.selected_i % height)
        sel_c = min(width - 1, state.selected_j % width)

        for r, c in points:
            if r < height and c < width:
                canvas[r][c] = "[cyan]•[/]"

        canvas[sel_r][sel_c] = "[black on bright_white]●[/]"

        out = Text()
        for row in canvas:
            line = Text()
            for cell in row:
                line.append_text(Text.from_markup(cell))
            out.append_text(line)
            out.append("\n")
        return out

    def render_from_state(self, state: ObservatoryState, mode_data=None, mds_data=None) -> None:
        if state.view_space == "mds":
            content = self._render_mds_real(state, mds_data)
        elif state.mode == "Run":
            content = self._render_run_grid(state, mode_data)
        elif state.mode == "Geometry":
            value_col = self._geometry_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.geometry_df if mode_data else None,
                value_col,
                signed=False,
            )
        elif state.mode == "Phase":
            value_col = self._phase_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.phase_df if mode_data else None,
                value_col,
                signed=(state.overlay == "signed_phase"),
            )
        elif state.mode == "Topology":
            value_col = self._topology_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.topology_df if mode_data else None,
                value_col,
                signed=False,
            )
        elif state.mode == "Operators":
            value_col = self._operators_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.operators_df if mode_data else None,
                value_col,
                signed=False,
            )
        elif state.mode == "Identity":
            value_col = self._identity_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.identity_nodes_df if mode_data else None,
                value_col,
                signed=(state.overlay in {"signed_local_obstruction", "legacy_spin"}),
            )
        else:
            content = self._render_run_grid(state, None)

        title = f"Manifold — {state.mode} / {state.view_space.upper()} / {state.overlay}"
        self.update(Panel(content, title=title, border_style="green"))