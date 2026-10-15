"""Stateless orchestration for the public RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..contracts import EmbeddingModel, LLM, Reranker, VectorStore
from ..models import Document
from ..registry import IntegrationRegistries

from .config import PipelineConfig


class RAGPipeline:
    """Coordinate retrieval and generation using injected components.

    The pipeline owns configuration, not conversation history. Callers pass
    history explicitly until the public session API is introduced.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        self._config = config or PipelineConfig()

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
        )

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
        if self._reranker is not None:
            documents = self._reranker.rerank(
                normalized_question,
                documents,
                self._config.top_k,
            )
        return {
            "question": self._build_prompt(normalized_question, documents),
            "history": [dict(message) for message in history],
        }

    @staticmethod
    def _build_prompt(question: str, documents: Sequence[Document]) -> str:
        context = "\n\n".join(document.page_content for document in documents)
        return (
            "Use only the retrieved context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
