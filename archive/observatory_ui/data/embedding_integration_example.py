from __future__ import annotations

from pathlib import Path

from pam.observatory.data.embedding_adapter import (
    EmbeddingAdapterConfig,
    load_embedding_points,
)


if __name__ == "__main__":
    r_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    alpha_values = [0.032, 0.048, 0.064, 0.080, 0.096]

    points = load_embedding_points(
        EmbeddingAdapterConfig(outputs_dir=Path("outputs")),
        r_values=r_values,
        alpha_values=alpha_values,
    )

    print("n_points:", len(points))
    if points:
        print("first:", points[0])
        print("last:", points[-1])
