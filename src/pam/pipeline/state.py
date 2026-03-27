from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pam.io.paths import ObservatoryPaths, OutputPaths


@dataclass
class PipelineState:
    """
    Shared file-first state for the canonical PAM pipeline.

    This indexes artifact families and roots. It does not replace files
    or attempt to store the scientific state in memory.
    """

    outputs: OutputPaths
    observatory: ObservatoryPaths
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_roots(
        cls,
        outputs_root: str | Path = "outputs",
        observatory_root: str | Path = "observatory",
        metadata: dict[str, Any] | None = None,
    ) -> "PipelineState":
        return cls(
            outputs=OutputPaths(Path(outputs_root)),
            observatory=ObservatoryPaths(Path(observatory_root)),
            metadata={} if metadata is None else dict(metadata),
        )

    def with_metadata(self, **updates: Any) -> "PipelineState":
        merged = dict(self.metadata)
        merged.update(updates)
        return PipelineState(
            outputs=self.outputs,
            observatory=self.observatory,
            metadata=merged,
        )
