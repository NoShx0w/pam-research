from __future__ import annotations

from textual.widget import Widget

from pam.observatory.renderers.trajectory_renderer import render_trajectory_panel


class TrajectoryWidget(Widget):
    def render(self):
        return render_trajectory_panel(self.app.state, self.app.trajectory_series)
