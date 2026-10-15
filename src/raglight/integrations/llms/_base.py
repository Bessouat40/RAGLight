"""Shared implementation details for built-in LLM integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class LLMAdapter:
    """Initialize a provider client without defining a public base class."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        system_prompt_file: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.system_prompt = (
            Path(system_prompt_file).read_text(encoding="utf-8")
            if system_prompt_file
            else system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        self.model: Any = self.load()

    def load(self) -> Any:
        raise NotImplementedError
