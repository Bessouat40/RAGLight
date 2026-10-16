"""Stateless orchestration for the public RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any
from uuid import uuid4

from ..contracts import EmbeddingModel, LLM, Reranker, VectorStore
from ..models import (
    Document,
    ExecutionEvent,
    FolderSource,
    IngestionResult,
    RAGResponse,
)
from ..registry import IntegrationRegistries

from .config import PipelineConfig
from .ingestion import DocumentProcessorRegistry, ingest_folders
from .session import ConversationSession


class RAGPipeline:
    """Coordinate retrieval and generation using injected components.

    The pipeline owns configuration, not conversation history. Callers either
    pass history explicitly or create an isolated conversation session.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        config: PipelineConfig | None = None,
        document_processors: DocumentProcessorRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        self._config = config or PipelineConfig()
        self._document_processors = document_processors

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        *,
        registries: IntegrationRegistries,
    ) -> RAGPipeline:
        """Construct a pipeline from names resolved by explicit registries."""

        required = {
            "llm": config.llm,
            "embeddings": config.embeddings,
            "vector_store": config.vector_store,
        }
        missing = [name for name, component in required.items() if component is None]
        if missing:
            raise ValueError(
                "registry-backed construction requires: " + ", ".join(missing)
            )

        llm_config = config.llm
        embeddings_config = config.embeddings
        vector_store_config = config.vector_store
        assert llm_config is not None
        assert embeddings_config is not None
        assert vector_store_config is not None

        embeddings = registries.embeddings.create(
            embeddings_config.name, **embeddings_config.options
        )
        vector_store_options = dict(vector_store_config.options)
        if "embeddings_model" in vector_store_options:
            raise ValueError(
                "vector_store options must not define reserved 'embeddings_model'"
            )
        vector_store = registries.vector_stores.create(
            vector_store_config.name,
            embeddings_model=embeddings,
            **vector_store_options,
        )
        llm = registries.llms.create(llm_config.name, **llm_config.options)
        reranker = (
            registries.rerankers.create(config.reranker.name, **config.reranker.options)
            if config.reranker is not None
            else None
        )
        return cls(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
            reranker=reranker,
            config=config,
            document_processors=registries.document_processors,
        )

    @property
    def config(self) -> PipelineConfig:
        """Return immutable pipeline behavior configuration."""

        return self._config

    def create_session(
        self,
        *,
        session_id: str | None = None,
        history: Sequence[Mapping[str, str]] = (),
    ) -> ConversationSession:
        """Create an isolated conversation over this shared pipeline."""

        return ConversationSession(self, session_id=session_id, history=history)

    def ingest(self, sources: Sequence[FolderSource]) -> IngestionResult:
        """Discover, process, and index files from one or more directories."""

        return ingest_folders(
            sources,
            vector_store=self._vector_store,
            processors=self._document_processors,
        )

    def generate(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> RAGResponse:
        """Retrieve context and return an observable structured response."""

        trace_id = str(uuid4())
        total_started = perf_counter()
        payload, documents, events, timings, retrieved_count = self._prepare_generation(
            question, history
        )

        generation_started = perf_counter()
        answer = self._llm.generate(payload)
        generation_duration = self._elapsed_ms(generation_started)
        timings["generation"] = generation_duration
        events.append(
            ExecutionEvent(
                stage="generation",
                status="completed",
                duration_ms=generation_duration,
                metrics={"answer_characters": len(answer)},
                message="Generated the final answer.",
            )
        )
        timings["total"] = self._elapsed_ms(total_started)
        return RAGResponse(
            answer=answer,
            sources=documents,
            events=events,
            timings=timings,
            metadata={
                "top_k": self._config.top_k,
                "retrieved_count": retrieved_count,
                "reranked": self._reranker is not None,
            },
            trace_id=trace_id,
        )

    def stream(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> Iterable[str]:
        """Retrieve context and yield generated text chunks in order."""

        payload, _, _, _, _ = self._prepare_generation(question, history)
        yield from self._llm.generate_streaming(payload)

    def _prepare_generation(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
    ) -> tuple[
        dict[str, Any],
        Sequence[Document],
        list[ExecutionEvent],
        dict[str, float],
        int,
    ]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        events: list[ExecutionEvent] = []
        timings: dict[str, float] = {}
        retrieval_started = perf_counter()
        documents = self._vector_store.similarity_search(
            normalized_question,
            k=self._config.top_k,
        )
        retrieval_duration = self._elapsed_ms(retrieval_started)
        retrieved_count = len(documents)
        timings["retrieval"] = retrieval_duration
        events.append(
            ExecutionEvent(
                stage="retrieval",
                status="completed",
                duration_ms=retrieval_duration,
                metrics={
                    "document_count": retrieved_count,
                    "requested_top_k": self._config.top_k,
                },
                message=f"Retrieved {retrieved_count} source document(s).",
            )
        )
        if self._reranker is not None:
            reranking_started = perf_counter()
            documents = self._reranker.rerank(
                normalized_question,
                documents,
                self._config.top_k,
            )
            reranking_duration = self._elapsed_ms(reranking_started)
            timings["reranking"] = reranking_duration
            events.append(
                ExecutionEvent(
                    stage="reranking",
                    status="completed",
                    duration_ms=reranking_duration,
                    metrics={"document_count": len(documents)},
                    message=f"Reranked {len(documents)} source document(s).",
                )
            )
        return (
            {
                "question": self._build_prompt(normalized_question, documents),
                "history": [dict(message) for message in history],
            },
            documents,
            events,
            timings,
            retrieved_count,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return max(0.0, (perf_counter() - started) * 1000)

    @staticmethod
    def _build_prompt(question: str, documents: Sequence[Document]) -> str:
        context = "\n\n".join(document.page_content for document in documents)
        return (
            "Use only the retrieved context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
