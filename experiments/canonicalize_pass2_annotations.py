from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.io.paths import ObservatoryPaths
from pam.pipeline.artifacts import mirror_file


@dataclass(frozen=True)
class Config:
    inputs_root: str | None = None


def resolve_input_path(value: str | None, default_value: str, inputs_root: str | None) -> str | None:
    if value is None:
        return None
    if inputs_root is None:
        return value
    if value != default_value:
        return value
    rel = Path(default_value)
    if rel.parts and rel.parts[0] == "outputs":
        rel = Path(*rel.parts[1:])
    return str(Path(inputs_root) / rel)



def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize pass2 annotations.")
    parser.add_argument(
        "--inputs-root",
        default=None,
        help=(
            "Optional campaign/pipeline root. When provided, default scale_root "
            "and outputs_root are resolved under this root."
        ),
    )
    args = parser.parse_args()
    cfg = Config(
        inputs_root=args.inputs_root,
    )

    observatory = ObservatoryPaths(Path("observatory"))

    mirror_file(
        Path(f"{cfg.inputs_root}/obs022_scene_bundle/scene_hubs.csv"),
        observatory.topology_hub_nodes_csv,
    )
    mirror_file(
        Path(f"{cfg.inputs_root}/obs024_family_hotspot_occupancy/family_hotspot_occupancy_nodes.csv"),
        observatory.topology_hotspot_nodes_csv,
    )
    mirror_file(
        Path(f"{cfg.inputs_root}/obs028c_canonical_seam_bundle/seam_nodes.csv"),
        observatory.topology_seam_bundle_nodes_csv,
    )
    mirror_file(
        Path(f"{cfg.inputs_root}/obs028c_canonical_seam_bundle/seam_embedding_summary.csv"),
        observatory.topology_seam_bundle_embedding_summary_csv,
    )
    mirror_file(
        Path(f"{cfg.inputs_root}/obs028c_canonical_seam_bundle/seam_family_summary.csv"),
        observatory.topology_seam_bundle_family_summary_csv,
    )

    print(observatory.topology_annotations_dir)


if __name__ == "__main__":
    main()
