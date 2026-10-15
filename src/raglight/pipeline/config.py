"""Provider-independent configuration for the standard RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Control standard retrieval behavior without selecting integrations."""

    top_k: int = 5

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
