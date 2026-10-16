"""Optional native Langfuse tracing integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID


class LangfuseTracer:
    """Map RAGLight observations to the current Langfuse Python SDK."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from langfuse import get_client
            except ModuleNotFoundError as error:
                from .._optional import optional_import_error

                optional_import_error(
                    error,
                    dependency="langfuse",
                    extra="langfuse",
                    feature="Langfuse observability",
                )
            client = get_client()
        self._client = client

    def observe(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ):
        options: dict[str, Any] = {
            "name": name,
            "as_type": as_type,
            "input": input,
            "metadata": dict(metadata) if metadata is not None else None,
        }
        if trace_id is not None:
            options["trace_context"] = {"trace_id": UUID(trace_id).hex}
        return self._client.start_as_current_observation(
            **{key: value for key, value in options.items() if value is not None}
        )

    def flush(self) -> None:
        """Flush queued observations in short-lived processes."""

        self._client.flush()


__all__ = ["LangfuseTracer"]
