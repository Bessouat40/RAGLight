"""Provider-independent configuration for the standard RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from collections.abc import Mapping
from typing import Any

from ..models import ComponentConfig


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Control retrieval and optionally select registered integrations."""

    top_k: int = 5
    reformulate_queries: bool = False
    llm: ComponentConfig | None = None
    embeddings: ComponentConfig | None = None
    vector_store: ComponentConfig | None = None
    reranker: ComponentConfig | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PipelineConfig:
        """Create validated pipeline configuration from serialized values."""

        fields = {
            "top_k",
            "reformulate_queries",
            "llm",
            "embeddings",
            "vector_store",
            "reranker",
        }
        unknown = set(value) - fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown pipeline configuration fields: {names}")

        def component(name: str) -> ComponentConfig | None:
            raw = value.get(name)
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise TypeError(f"{name} configuration must be a mapping")
            return ComponentConfig.from_dict(raw)

        return cls(
            top_k=int(value.get("top_k", 5)),
            reformulate_queries=bool(value.get("reformulate_queries", False)),
            llm=component("llm"),
            embeddings=component("embeddings"),
            vector_store=component("vector_store"),
            reranker=component("reranker"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load a JSON or TOML pipeline configuration."""

        config_path = Path(path)
        if config_path.suffix.lower() == ".json":
            value = json.loads(config_path.read_text(encoding="utf-8"))
        elif config_path.suffix.lower() == ".toml":
            with config_path.open("rb") as config_file:
                value = tomllib.load(config_file)
        else:
            raise ValueError("configuration file must use .json or .toml")
        if not isinstance(value, Mapping):
            raise TypeError("pipeline configuration must be a mapping")
        return cls.from_dict(value)
