"""Stateless orchestration for the public RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from time import perf_counter
from typing import Any
from uuid import uuid4

from ..contracts import EmbeddingModel, LLM, Reranker, Tracer, VectorStore
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
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        self._config = config or PipelineConfig()
        self._document_processors = document_processors
        self._tracer = tracer

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        *,
        registries: IntegrationRegistries,
        tracer: Tracer | None = None,
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
            tracer=tracer,
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
        with self._observe(
            name="raglight.rag",
            as_type="chain",
            input={"question": question, "history": list(history)},
            metadata={"top_k": self._config.top_k},
            trace_id=trace_id,
        ) as trace:
            payload, documents, events, timings, retrieved_count, retrieval_query = (
                self._prepare_generation(question, history)
            )

            generation_started = perf_counter()
            with self._observe(
                name="raglight.generation",
                as_type="generation",
                input=payload,
            ) as generation:
                answer = self._llm.generate(payload)
                generation.update(output=answer)
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
            metadata: dict[str, Any] = {
                "top_k": self._config.top_k,
                "retrieved_count": retrieved_count,
                "reranked": self._reranker is not None,
            }
            if retrieval_query != question.strip():
                metadata["retrieval_query"] = retrieval_query
            response = RAGResponse(
                answer=answer,
                sources=documents,
                events=events,
                timings=timings,
                metadata=metadata,
                trace_id=trace_id,
            )
            trace.update(
                output={"answer": answer},
                metadata={"retrieved_count": retrieved_count},
            )
            return response

    def stream(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> Iterable[str]:
        """Retrieve context and yield generated text chunks in order."""

        trace_id = str(uuid4())
        with self._observe(
            name="raglight.rag.stream",
            as_type="chain",
            input={"question": question, "history": list(history)},
            metadata={"top_k": self._config.top_k},
            trace_id=trace_id,
        ) as trace:
            payload, _, _, _, _, _ = self._prepare_generation(question, history)
            chunks: list[str] = []
            with self._observe(
                name="raglight.generation",
                as_type="generation",
                input=payload,
            ) as generation:
                for chunk in self._llm.generate_streaming(payload):
                    chunks.append(chunk)
                    yield chunk
                answer = "".join(chunks)
                generation.update(output=answer)
            trace.update(output={"answer": answer})

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
        str,
    ]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        events: list[ExecutionEvent] = []
        timings: dict[str, float] = {}
        normalized_history = [dict(message) for message in history]
        retrieval_query = normalized_question
        if self._config.reformulate_queries and normalized_history:
            reformulation_started = perf_counter()
            reformulation_input = {
                "question": self._build_reformulation_prompt(
                    normalized_question, normalized_history
                ),
                "history": [],
            }
            with self._observe(
                name="raglight.reformulation",
                as_type="generation",
                input=reformulation_input,
            ) as reformulation:
                retrieval_query = self._llm.generate(reformulation_input).strip()
                if not retrieval_query:
                    raise ValueError("query reformulation returned an empty query")
                reformulation.update(output=retrieval_query)
            reformulation_duration = self._elapsed_ms(reformulation_started)
            timings["reformulation"] = reformulation_duration
            events.append(
                ExecutionEvent(
                    stage="reformulation",
                    status="completed",
                    duration_ms=reformulation_duration,
                    message="Rewrote the conversational query for retrieval.",
                )
            )
        retrieval_started = perf_counter()
        with self._observe(
            name="raglight.retrieval",
            as_type="retriever",
            input={"query": retrieval_query, "top_k": self._config.top_k},
        ) as retrieval:
            documents = self._vector_store.similarity_search(
                retrieval_query,
                k=self._config.top_k,
            )
            retrieval.update(output=self._documents_payload(documents))
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
            with self._observe(
                name="raglight.reranking",
                as_type="span",
                input=self._documents_payload(documents),
            ) as reranking:
                documents = self._reranker.rerank(
                    retrieval_query,
                    documents,
                    self._config.top_k,
                )
                reranking.update(output=self._documents_payload(documents))
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
                "history": normalized_history,
            },
            documents,
            events,
            timings,
            retrieved_count,
            retrieval_query,
        )

    def _observe(self, **values: Any) -> AbstractContextManager[Any]:
        if self._tracer is None:
            return nullcontext(_NullObservation())
        return self._tracer.observe(**values)

    @staticmethod
    def _documents_payload(documents: Sequence[Document]) -> list[dict[str, Any]]:
        return [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in documents
        ]

    @staticmethod
    def _build_reformulation_prompt(
        question: str,
        history: Sequence[Mapping[str, str]],
    ) -> str:
        transcript = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in history
        )
        return (
            "Rewrite the follow-up question as one standalone retrieval query. "
            "Preserve names, constraints, and intent. Return only the query.\n\n"
            f"Conversation:\n{transcript}\n\n"
            f"Follow-up question:\n{question}"
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


class _NullObservation:
    def update(self, **values: Any) -> None:
        del values
