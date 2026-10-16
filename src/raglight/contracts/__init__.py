"""Stable structural contracts for extending RAGLight.

Implementations conform by providing the documented methods. They do not need
to inherit from these protocols or from any RAGLight base class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from ..models import Document, RetrievalFilter


@runtime_checkable
class LLM(Protocol):
    """Generate complete or streamed text from a RAGLight input payload."""

    def generate(self, input: dict[str, Any]) -> str:
        """Return a complete answer for the supplied generation input."""

        ...

    def generate_streaming(
        self,
        input: dict[str, Any],
    ) -> Iterable[str]:
        """Yield answer chunks in generation order."""

        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Create vector representations for documents and queries."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts in input order."""

        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""

        ...


@runtime_checkable
class VectorStore(Protocol):
    """Retrieve documents relevant to a question."""

    def similarity_search(
        self,
        question: str,
        k: int = 5,
        *,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[Document]:
        """Return relevant documents inside the optional metadata scope."""

        ...


@runtime_checkable
class IngestionVectorStore(VectorStore, Protocol):
    """Retrieve documents and persist processed document chunks."""

    def add_documents(self, documents: list[Document]) -> None:
        """Persist document chunks for subsequent retrieval."""

        ...


@runtime_checkable
class Reranker(Protocol):
    """Reorder retrieved documents for a question."""

    def rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Return at most ``top_k`` documents in descending relevance order."""

        ...


@runtime_checkable
class DocumentProcessor(Protocol):
    """Turn one source file into document and class chunks."""

    def process(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, list[Document]]:
        """Return ``chunks`` and ``classes`` document collections."""

        ...


@runtime_checkable
class TraceObservation(Protocol):
    """Update one active provider-neutral tracing observation."""

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> Any: ...


@runtime_checkable
class Tracer(Protocol):
    """Create nested observations without coupling pipelines to a vendor SDK."""

    def observe(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AbstractContextManager[TraceObservation]: ...


__all__ = [
    "DocumentProcessor",
    "EmbeddingModel",
    "IngestionVectorStore",
    "LLM",
    "Reranker",
    "TraceObservation",
    "Tracer",
    "VectorStore",
]
