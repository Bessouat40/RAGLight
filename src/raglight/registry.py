"""Provider-independent registry primitives used by optional integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Generic, TypeVar, cast

ComponentT = TypeVar("ComponentT")
Factory = Callable[..., ComponentT]


@dataclass(frozen=True, slots=True)
class _LazyFactory(Generic[ComponentT]):
    module: str
    attribute: str

    def resolve(self) -> Factory[ComponentT]:
        value = getattr(import_module(self.module), self.attribute)
        if not callable(value):
            raise TypeError(
                f"Registered integration {self.module}.{self.attribute} "
                "is not callable"
            )
        return cast(Factory[ComponentT], value)


class IntegrationRegistry(Generic[ComponentT]):
    """Map stable names to component factories without importing providers."""

    def __init__(self, category: str) -> None:
        normalized_category = category.strip()
        if not normalized_category:
            raise ValueError("registry category must not be empty")
        self._category = normalized_category
        self._entries: dict[str, Factory[ComponentT] | _LazyFactory[ComponentT]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(sorted(self._entries))

    def register(
        self,
        name: str,
        factory: Factory[ComponentT],
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom component factory."""

        if not callable(factory):
            raise TypeError("integration factory must be callable")
        self._set(name, factory, replace=replace)

    def register_lazy(
        self,
        name: str,
        module: str,
        attribute: str,
        *,
        replace: bool = False,
    ) -> None:
        """Register a factory by import path and resolve it only when created."""

        if not module.strip() or not attribute.strip():
            raise ValueError("lazy integration paths must not be empty")
        self._set(
            name,
            _LazyFactory(module.strip(), attribute.strip()),
            replace=replace,
        )

    def create(self, name: str, /, **options: Any) -> ComponentT:
        """Create a registered component with provider-owned options."""

        normalized = self._normalize_name(name)
        try:
            entry = self._entries[normalized]
        except KeyError as error:
            available = ", ".join(self.names) or "none"
            raise LookupError(
                f"Unknown {self._category} integration '{normalized}'. "
                f"Available integrations: {available}"
            ) from error
        factory = entry.resolve() if isinstance(entry, _LazyFactory) else entry
        return factory(**options)

    def _set(
        self,
        name: str,
        entry: Factory[ComponentT] | _LazyFactory[ComponentT],
        *,
        replace: bool,
    ) -> None:
        normalized = self._normalize_name(name)
        if normalized in self._entries and not replace:
            raise ValueError(
                f"{self._category} integration '{normalized}' is already registered"
            )
        self._entries[normalized] = entry

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("integration name must not be empty")
        return normalized


@dataclass(frozen=True, slots=True)
class IntegrationRegistries:
    """Group registries used to construct a complete RAG pipeline."""

    llms: IntegrationRegistry[Any]
    embeddings: IntegrationRegistry[Any]
    vector_stores: IntegrationRegistry[Any]
    rerankers: IntegrationRegistry[Any]
    document_processors: IntegrationRegistry[Any]
