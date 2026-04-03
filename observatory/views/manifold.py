from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

from observatory.state import ObservatoryState
from observatory.views.formatting import mode_label, overlay_label
from observatory.views.scalars import render_signed_cell, render_unsigned_cell


class ManifoldView(Static):
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

    def _coord_key(self, r, a) -> tuple[float, float]:
        return (round(float(r), 9), round(float(a), 9))

    def _lookup_from_df(self, df, value_col: str, grid_r_vals, grid_a_vals):
        lookup: dict[tuple[int, int], object] = {}

        if (
            df is not None
            and not df.empty
            and {"r", "alpha", value_col}.issubset(df.columns)
            and grid_r_vals is not None
            and grid_a_vals is not None
        ):
            value_map: dict[tuple[float, float], object] = {}
            for _, row in df.iterrows():
                try:
                    key = self._coord_key(row["r"], row["alpha"])
                    value_map[key] = row[value_col]
                except Exception:
                    continue

            for i, r in enumerate(grid_r_vals):
                for j, a in enumerate(grid_a_vals):
                    lookup[(i, j)] = value_map.get(self._coord_key(r, a), None)

        return lookup

    def _marker_symbol(self) -> str:
        return "[yellow]✦[/]"

    def _render_grid_blocks(
        self,
        state: ObservatoryState,
        lookup: dict[tuple[int, int], object],
        *,
        signed: bool,
        grid_r_vals=None,
        grid_a_vals=None,
        marker_coords=None,
    ) -> Text:
        width, height = self._drawable_size()
        width = max(12, width)
        height = max(8, height)

        values = [v for v in lookup.values() if v is not None]
        if not values:
            values = [0.0]

        if signed:
            vabs = max(abs(float(v)) for v in values)
        else:
            vmin = min(float(v) for v in values)
            vmax = max(float(v) for v in values)

        canvas = [[" " for _ in range(width)] for _ in range(height)]

        def x_bounds(j: int) -> tuple[int, int]:
            x0 = int(j * width / state.grid_cols)
            x1 = int((j + 1) * width / state.grid_cols)
            return x0, max(x0 + 1, x1)

        def y_bounds(i: int) -> tuple[int, int]:
            y0 = int(i * height / state.grid_rows)
            y1 = int((i + 1) * height / state.grid_rows)
            return y0, max(y0 + 1, y1)

        for i in range(state.grid_rows):
            for j in range(state.grid_cols):
                val = lookup.get((i, j), None)

                if signed:
                    cell_markup = render_signed_cell(val, vabs=vabs, selected=False)
                else:
                    cell_markup = render_unsigned_cell(val, vmin=vmin, vmax=vmax, selected=False)

                x0, x1 = x_bounds(j)
                y0, y1 = y_bounds(i)

                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        canvas[yy][xx] = cell_markup

                is_marker = False
                if (
                    marker_coords is not None
                    and grid_r_vals is not None
                    and grid_a_vals is not None
                    and i < len(grid_r_vals)
                    and j < len(grid_a_vals)
                ):
                    key = (round(float(grid_r_vals[i]), 9), round(float(grid_a_vals[j]), 9))
                    is_marker = key in marker_coords

                if is_marker:
                    cy = (y0 + y1 - 1) // 2
                    cx = (x0 + x1 - 1) // 2
                    canvas[cy][cx] = self._marker_symbol()

                if i == state.selected_i and j == state.selected_j:
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            canvas[yy][xx] = "[#d7d7d7 on #444444] [/]"

                    cy = (y0 + y1 - 1) // 2
                    cx = (x0 + x1 - 1) // 2
                    canvas[cy][cx] = "[black on bright_white]●[/]"

        out = Text()
        for row in canvas:
            line = Text()
            for cell in row:
                line.append_text(Text.from_markup(cell))
            out.append_text(line)
            out.append("\n")
        return out

    def _render_run_grid(self, state: ObservatoryState, run_data, *, grid_r_vals=None, grid_a_vals=None, marker_coords=None) -> Text:
        coverage_lookup: dict[tuple[int, int], int] = {}

        if run_data is not None and not run_data.coverage_df.empty:
            r_vals = sorted(run_data.coverage_df["r"].dropna().unique())
            a_vals = sorted(run_data.coverage_df["alpha"].dropna().unique())
            for i, r in enumerate(r_vals):
                for j, a in enumerate(a_vals):
                    hit = run_data.coverage_df[
                        (run_data.coverage_df["r"] == r) & (run_data.coverage_df["alpha"] == a)
                    ]
                    coverage_lookup[(i, j)] = 0 if hit.empty else int(hit.iloc[0]["n_rows"])

        return self._render_grid_blocks(
            state,
            coverage_lookup,
            signed=False,
            grid_r_vals=grid_r_vals,
            grid_a_vals=grid_a_vals,
            marker_coords=marker_coords,
        )

    def _render_scalar_grid(
        self,
        state: ObservatoryState,
        df,
        value_col: str,
        *,
        signed: bool,
        grid_r_vals=None,
        grid_a_vals=None,
        marker_coords=None,
    ) -> Text:
        lookup = self._lookup_from_df(df, value_col, grid_r_vals, grid_a_vals)
        return self._render_grid_blocks(
            state,
            lookup,
            signed=signed,
            grid_r_vals=grid_r_vals,
            grid_a_vals=grid_a_vals,
            marker_coords=marker_coords,
        )

    def _render_mds_real(
        self,
        state: ObservatoryState,
        mds_data,
        grid_r_vals=None,
        grid_a_vals=None,
        marker_coords=None,
    ) -> Text:
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
            return pad_x + int(round((x_max - x) / (x_max - x_min) * (inner_w - 1)))

        def scale_y(y: float) -> int:
            if y_max == y_min:
                return pad_y + inner_h // 2
            return pad_y + int(round((y - y_min) / (y_max - y_min) * (inner_h - 1)))

        canvas = [[" " for _ in range(width)] for _ in range(height)]

        sel_key = None
        if (
            grid_r_vals is not None
            and grid_a_vals is not None
            and state.selected_i < len(grid_r_vals)
            and state.selected_j < len(grid_a_vals)
        ):
            sel_key = self._coord_key(
                grid_r_vals[state.selected_i],
                grid_a_vals[state.selected_j],
            )

        for _, row in df.iterrows():
            cx = scale_x(float(row["mds1"]))
            cy = scale_y(float(row["mds2"]))
            rr = height - 1 - cy

            if 0 <= rr < height and 0 <= cx < width:
                is_selected = (
                    sel_key is not None
                    and self._coord_key(row["r"], row["alpha"]) == sel_key
                )
                canvas[rr][cx] = "[black on bright_white]●[/]" if is_selected else "[cyan]•[/]"

            row_key = (round(float(row["r"]), 9), round(float(row["alpha"]), 9))
            is_marker = marker_coords is not None and row_key in marker_coords
            if is_selected:
                canvas[rr][cx] = "[black on bright_white]●[/]"
            elif is_marker:
                canvas[rr][cx] = self._marker_symbol()
            else:
                canvas[rr][cx] = "[cyan]•[/]"

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

    def render_from_state(
        self,
        state: ObservatoryState,
        mode_data=None,
        mds_data=None,
        grid_r_vals=None,
        grid_a_vals=None,
        marker_coords=None,
    ) -> None:
        if state.view_space == "mds":
            content = self._render_mds_real(
                state,
                mds_data,
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        elif state.mode == "Run":
            content = self._render_run_grid(
                state,
                mode_data,
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        elif state.mode == "Geometry":
            value_col = self._geometry_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.geometry_df if mode_data else None,
                value_col,
                signed=False,
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        elif state.mode == "Phase":
            value_col = self._phase_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.phase_df if mode_data else None,
                value_col,
                signed=(state.overlay == "signed_phase"),
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        elif state.mode == "Topology":
            value_col = self._topology_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.topology_df if mode_data else None,
                value_col,
                signed=False,
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        elif state.mode == "Operators":
            value_col = self._operators_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.operators_df if mode_data else None,
                value_col,
                signed=False,
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        elif state.mode == "Identity":
            value_col = self._identity_value_col(state.overlay)
            content = self._render_scalar_grid(
                state,
                mode_data.identity_nodes_df if mode_data else None,
                value_col,
                signed=(state.overlay in {"signed_local_obstruction", "legacy_spin"}),
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )
        else:
            content = self._render_run_grid(
                state,
                None,
                grid_r_vals=grid_r_vals,
                grid_a_vals=grid_a_vals,
                marker_coords=marker_coords,
            )

        title = f"Manifold — {mode_label(state.mode)} / {state.view_space.upper()} / {overlay_label(state.overlay)}"
        self.update(Panel(content, title=title, border_style="green"))