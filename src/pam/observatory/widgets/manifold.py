from textual.widget import Widget
from pam.observatory.renderers.manifold_renderer import render_parameter_manifold_panel

class ParameterManifoldWidget(Widget):
    def render(self):
        return render_parameter_manifold_panel(self.app.state)