"""Tool adapters, registry, and the built-in retrieval capability."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
import inspect
from typing import Any

from ..contracts import Reranker, VectorStore
from ..models import MatchNone, RetrievalFilter
from ..models.retrieval_filter import normalize_retrieval_filter
from .contracts import AgentTool
from .models import ToolResult, ToolSpec

ToolHandlerResult = str | ToolResult
ToolHandler = Callable[
    [Mapping[str, Any]], ToolHandlerResult | Awaitable[ToolHandlerResult]
]


class FunctionTool:
    """Expose a Python callable through the stable agent-tool contract."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: Mapping[str, Any] | None = None,
    ) -> None:
        spec = ToolSpec(name, description, input_schema or {})
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self.name = spec.name
        self.description = spec.description
        self.input_schema = spec.input_schema
        self._handler = handler

    def run(self, arguments: Mapping[str, Any]) -> ToolResult | Awaitable[ToolResult]:
        value = self._handler(arguments)
        if inspect.isawaitable(value):

            async def normalized() -> ToolResult:
                return self._normalize(await value)

            return normalized()
        return self._normalize(value)

    @staticmethod
    def _normalize(value: ToolHandlerResult) -> ToolResult:
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, str):
            return ToolResult(value)
        raise TypeError("tool handlers must return str or ToolResult")


class ToolRegistry:
    """Hold ready tool instances under deterministic normalized names."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(tool.name, tool.description, tool.input_schema)
            for tool in (self._tools[name] for name in self.names)
        )

    def register(self, tool: AgentTool) -> None:
        if not isinstance(tool, AgentTool):
            raise TypeError("agent tools must implement the AgentTool protocol")
        spec = ToolSpec(tool.name, tool.description, tool.input_schema)
        if spec.name in self._tools:
            raise ValueError(f"agent tool '{spec.name}' is already registered")
        self._tools[spec.name] = tool

    def get(self, name: str) -> AgentTool:
        normalized = name.strip().lower()
        try:
            return self._tools[normalized]
        except KeyError as error:
            available = ", ".join(self.names) or "none"
            raise LookupError(
                f"Unknown agent tool '{normalized}'. Available tools: {available}"
            ) from error


class RetrievalTool:
    """Expose vector retrieval and optional reranking as one agent tool."""

    name = "retrieve"
    description = "Retrieve relevant knowledge-base documents for a query."
    input_schema: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        top_k: int,
        reranker: Reranker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._reranker = reranker

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        return self.run_scoped(arguments)

    def run_scoped(
        self,
        arguments: Mapping[str, Any],
        *,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("retrieve requires a non-empty string 'query'")
        requested_top_k = arguments.get("top_k", self._top_k)
        if not isinstance(requested_top_k, int) or isinstance(requested_top_k, bool):
            raise ValueError("retrieve 'top_k' must be an integer")
        if requested_top_k < 1:
            raise ValueError("retrieve 'top_k' must be greater than zero")
        effective_top_k = min(requested_top_k, self._top_k)
        normalized_query = query.strip()
        normalized_filter = (
            normalize_retrieval_filter(retrieval_filter)
            if retrieval_filter is not None
            else None
        )
        if isinstance(normalized_filter, MatchNone):
            documents = []
        elif normalized_filter is None:
            documents = self._vector_store.similarity_search(
                normalized_query,
                k=effective_top_k,
            )
        else:
            documents = self._vector_store.similarity_search(
                normalized_query,
                k=effective_top_k,
                retrieval_filter=normalized_filter,
            )
        retrieved_count = len(documents)
        if self._reranker is not None:
            documents = self._reranker.rerank(
                normalized_query,
                documents,
                effective_top_k,
            )
        content = "\n\n".join(
            f"[{index}] {document.page_content}"
            for index, document in enumerate(documents, start=1)
        )
        return ToolResult(
            content=content,
            sources=documents,
            metadata={
                "query": normalized_query,
                "requested_top_k": effective_top_k,
                "retrieved_count": retrieved_count,
                "result_count": len(documents),
                "reranked": self._reranker is not None,
            },
        )
