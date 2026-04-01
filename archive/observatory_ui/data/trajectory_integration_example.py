from __future__ import annotations

from pathlib import Path

from pam.observatory.data.trajectory_adapter import (
    TrajectoryAdapterConfig,
    load_trajectory_series,
)


if __name__ == "__main__":
    series = load_trajectory_series(
        TrajectoryAdapterConfig(
            trajectories_dir=Path("outputs/trajectories"),
            max_points=48,
        ),
        r=0.15,
        alpha=0.064,
    )

    print("source_kind:", series.source_kind)
    print("source_path:", series.source_path)
    print("F_raw[:8]:", series.f_raw[:8])
    print("H_joint[:8]:", series.h_joint[:8])
    print("K[:8]:", series.k_series[:8])
    print("πF_smooth[:8]:", series.pif_smooth[:8])
