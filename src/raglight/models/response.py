"""Provider-independent response and execution-event values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Literal

from .document import Document

ExecutionStatus = Literal["started", "completed", "failed"]
_EVENT_STATUSES = frozenset({"started", "completed", "failed"})


def _immutable_mapping(
    value: Mapping[str, Any], *, field_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Describe one observable pipeline stage without exposing raw logs."""

    stage: str
    status: ExecutionStatus
    duration_ms: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stage = self.stage.strip()
        if not stage:
            raise ValueError("stage must not be empty")
        status = self.status.strip().lower()
        if status not in _EVENT_STATUSES:
            raise ValueError(f"unsupported event status: {self.status}")
        duration_ms = float(self.duration_ms)
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("duration_ms must not be negative or non-finite")

        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(
            self,
            "metrics",
            _immutable_mapping(self.metrics, field_name="metrics"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, field_name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a transport-friendly representation of the event."""

        return {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metrics": dict(self.metrics),
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Return an answer together with its sources and execution details."""

    answer: str
    trace_id: str
    sources: Sequence[Document] = ()
    events: Sequence[ExecutionEvent] = ()
    timings: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_id = self.trace_id.strip()
        if not trace_id:
            raise ValueError("trace_id must not be empty")
        timings = {name: float(value) for name, value in self.timings.items()}
        if any(not math.isfinite(value) or value < 0 for value in timings.values()):
            raise ValueError("timings must not be negative or non-finite")

        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "timings", MappingProxyType(timings))
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, field_name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for delivery layers."""

        return {
            "answer": self.answer,
            "sources": [
                {
                    "page_content": source.page_content,
                    "metadata": dict(source.metadata),
                }
                for source in self.sources
            ],
            "events": [event.to_dict() for event in self.events],
            "timings": dict(self.timings),
            "metadata": dict(self.metadata),
            "trace_id": self.trace_id,
        }
