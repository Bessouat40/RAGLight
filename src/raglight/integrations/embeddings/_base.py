"""Shared implementation details for built-in embedding integrations."""

from __future__ import annotations

from typing import Any


class EmbeddingAdapter:
    """Initialize a provider client without defining a public base class."""

    def __init__(self, model_name: str, api_base: str | None = None) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.model: Any = self.load()

    def load(self) -> Any:
        raise NotImplementedError
