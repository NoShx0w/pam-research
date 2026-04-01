from pathlib import Path

from pam.observatory.app import PamObservatory
from pam.observatory.data.adapter import AdapterConfig, load_state_from_index


def build_app_from_outputs(outputs_dir: str = "outputs") -> PamObservatory:
    app = PamObservatory()
    app.state = load_state_from_index(
        AdapterConfig(index_csv=Path(outputs_dir) / "index.csv", dataset_total=750)
    )
    return app


if __name__ == "__main__":
    build_app_from_outputs().run()
