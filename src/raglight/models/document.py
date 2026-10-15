"""Provider-independent document values used by the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A text fragment and its provider-independent metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
