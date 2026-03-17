from __future__ import annotations

from textual.widget import Widget

from pam.observatory.renderers.embedding_renderer import render_embedding_panel


class EmbeddingWidget(Widget):
    def render(self):
        return render_embedding_panel(self.app.state, self.app.embedding_points)
