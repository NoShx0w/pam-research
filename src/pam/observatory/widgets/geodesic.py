from __future__ import annotations

from textual.widget import Widget

from pam.observatory.renderers.geodesic_renderer import render_geodesic_panel


class GeodesicWidget(Widget):
    def render(self):
        probe = getattr(self.app, "geodesic_probe", None)
        return render_geodesic_panel(self.app.state, probe)