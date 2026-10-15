"""Serializable provider selection for registry-backed construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    """Select one registered component and pass options to its factory."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.name.strip().lower()
        if not normalized_name:
            raise ValueError("component name must not be empty")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentConfig:
        """Validate a serialized component configuration."""

        unknown = set(value) - {"name", "options"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown component configuration fields: {names}")
        if "name" not in value:
            raise ValueError("component configuration requires 'name'")
        options = value.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("component options must be a mapping")
        return cls(name=str(value["name"]), options=options)
