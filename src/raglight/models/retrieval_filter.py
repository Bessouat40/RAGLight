"""Provider-independent values for constraining document retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import TypeAlias

FilterScalar: TypeAlias = str | int | float | bool


def _field(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("field must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("field must not be empty")
    return normalized


def _scalar(value: object) -> FilterScalar:
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError("filter values must be scalar strings, numbers, or booleans")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("filter values must be finite")
    return value


@dataclass(frozen=True, slots=True)
class Eq:
    """Match metadata whose field equals one scalar value."""

    field: str
    value: FilterScalar

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _field(self.field))
        object.__setattr__(self, "value", _scalar(self.value))


@dataclass(frozen=True, slots=True, init=False)
class In:
    """Match metadata whose field equals any supplied scalar value."""

    field: str
    values: tuple[FilterScalar, ...]

    def __init__(self, field: str, values: Iterable[object]) -> None:
        if isinstance(values, (str, bytes)):
            raise TypeError("values must be an iterable of scalar values")
        try:
            normalized_values = tuple(_scalar(value) for value in values)
        except TypeError as error:
            if "not iterable" in str(error):
                raise TypeError(
                    "values must be an iterable of scalar values"
                ) from error
            raise
        if not normalized_values:
            raise ValueError("In requires at least one value")
        object.__setattr__(self, "field", _field(field))
        object.__setattr__(self, "values", normalized_values)


@dataclass(frozen=True, slots=True)
class MatchNone:
    """Match no documents without issuing an unrestricted provider query."""


@dataclass(frozen=True, slots=True, init=False)
class And:
    """Require every child retrieval filter to match."""

    filters: tuple[RetrievalFilter, ...]

    def __init__(self, *filters: RetrievalFilter) -> None:
        object.__setattr__(self, "filters", _filters("And", filters))


@dataclass(frozen=True, slots=True, init=False)
class Or:
    """Require at least one child retrieval filter to match."""

    filters: tuple[RetrievalFilter, ...]

    def __init__(self, *filters: RetrievalFilter) -> None:
        object.__setattr__(self, "filters", _filters("Or", filters))


RetrievalFilter: TypeAlias = Eq | In | MatchNone | And | Or
_FILTER_TYPES = (Eq, In, MatchNone, And, Or)


def _filters(
    operator: str, filters: tuple[RetrievalFilter, ...]
) -> tuple[RetrievalFilter, ...]:
    if not filters:
        raise ValueError(f"{operator} requires at least one filter")
    if any(not isinstance(item, _FILTER_TYPES) for item in filters):
        raise TypeError(f"{operator} children must be retrieval filters")
    return tuple(filters)


def normalize_retrieval_filter(value: RetrievalFilter) -> RetrievalFilter:
    """Simplify logical no-match branches before provider translation."""

    if isinstance(value, (Eq, In, MatchNone)):
        return value
    if not isinstance(value, (And, Or)):
        raise TypeError("retrieval_filter must be a RAGLight retrieval filter")
    children = tuple(normalize_retrieval_filter(child) for child in value.filters)
    if isinstance(value, And):
        if any(isinstance(child, MatchNone) for child in children):
            return MatchNone()
        return children[0] if len(children) == 1 else And(*children)
    retained = tuple(child for child in children if not isinstance(child, MatchNone))
    if not retained:
        return MatchNone()
    return retained[0] if len(retained) == 1 else Or(*retained)
