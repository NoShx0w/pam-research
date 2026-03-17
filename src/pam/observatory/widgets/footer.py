from textual.widget import Widget
from rich.text import Text

class FooterWidget(Widget):
    def render(self):
        s=self.app.state
        return Text(
            f"r={s.selected_r:.2f} α={s.selected_alpha:.3f} | "
            "↑↓←→ move | m manifold | e embed | g probe | q quit"
        )