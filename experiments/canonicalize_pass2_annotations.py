from __future__ import annotations

from pathlib import Path

from pam.io.paths import ObservatoryPaths
from pam.pipeline.artifacts import mirror_file


def main() -> None:
    observatory = ObservatoryPaths(Path("observatory"))

    mirror_file(
        Path("outputs/obs022_scene_bundle/scene_hubs.csv"),
        observatory.topology_hub_nodes_csv,
    )
    mirror_file(
        Path("outputs/obs024_family_hotspot_occupancy/family_hotspot_occupancy_nodes.csv"),
        observatory.topology_hotspot_nodes_csv,
    )
    mirror_file(
        Path("outputs/obs028c_canonical_seam_bundle/seam_nodes.csv"),
        observatory.topology_seam_bundle_nodes_csv,
    )
    mirror_file(
        Path("outputs/obs028c_canonical_seam_bundle/seam_embedding_summary.csv"),
        observatory.topology_seam_bundle_embedding_summary_csv,
    )
    mirror_file(
        Path("outputs/obs028c_canonical_seam_bundle/seam_family_summary.csv"),
        observatory.topology_seam_bundle_family_summary_csv,
    )

    print(observatory.topology_annotations_dir)


if __name__ == "__main__":
    main()
