from textual.widget import Widget
from rich.text import Text

class StatusWidget(Widget):
    def render(self):
        s=self.app.state
        return Text(f"PAM OBSERVATORY   dataset {s.dataset_progress}/{s.dataset_total}")