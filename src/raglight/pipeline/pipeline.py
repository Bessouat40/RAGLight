"""Stateless orchestration for the public RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from .config import PipelineConfig


class _Document(Protocol):
    page_content: str


class _LLM(Protocol):
    def generate(self, input: dict[str, Any]) -> str: ...

    def generate_streaming(
        self, input: dict[str, Any], callbacks: list[Any] | None = None
    ) -> Iterable[str]: ...


class _EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class _VectorStore(Protocol):
    def similarity_search(self, question: str, k: int = 5) -> list[_Document]: ...


class RAGPipeline:
    """Coordinate retrieval and generation using injected components.

    The pipeline owns configuration, not conversation history. Callers pass
    history explicitly until the public session API is introduced.
    """

    def __init__(
        self,
        *,
        llm: _LLM,
        embeddings: _EmbeddingModel,
        vector_store: _VectorStore,
        config: PipelineConfig | None = None,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._config = config or PipelineConfig()

    @property
    def config(self) -> PipelineConfig:
        """Return immutable pipeline behavior configuration."""

        return self._config

    def generate(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> str:
        """Retrieve context and return one complete generated answer."""

        payload = self._generation_input(question, history)
        return self._llm.generate(payload)

    def stream(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> Iterable[str]:
        """Retrieve context and yield generated text chunks in order."""

        payload = self._generation_input(question, history)
        yield from self._llm.generate_streaming(payload)

    def _generation_input(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        documents = self._vector_store.similarity_search(
            normalized_question,
            k=self._config.top_k,
        )
        return {
            "question": self._build_prompt(normalized_question, documents),
            "history": [dict(message) for message in history],
        }

    @staticmethod
    def _build_prompt(question: str, documents: Sequence[_Document]) -> str:
        context = "\n\n".join(document.page_content for document in documents)
        return (
            "Use only the retrieved context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
