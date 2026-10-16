from __future__ import annotations

from pathlib import Path
from typing import Any, List, Dict, Optional
import logging

from ...contracts import EmbeddingModel
from ...models import Document, MatchNone, RetrievalFilter
from ...models.retrieval_filter import normalize_retrieval_filter
from .bm25_index import BM25Index


class VectorStoreAdapter:
    """
    Abstract base class for vector store implementations.

    This class provides shared BM25/hybrid search logic and defines the
    storage methods that concrete implementations must provide.
    """

    def __init__(
        self,
        persist_directory: str,
        embeddings_model: EmbeddingModel,
        search_type: str = "semantic",
        alpha: float = 0.5,
    ) -> None:
        self.embeddings_model = embeddings_model
        self.persist_directory: str = persist_directory
        self.vector_store: Any = None
        self.vector_store_classes: Any = None
        self.search_type = search_type
        self.alpha = alpha
        self._bm25 = BM25Index()

    # ------------------------------------------------------------------
    # BM25 / hybrid helpers (shared across all backends)
    # ------------------------------------------------------------------

    def _bm25_path(self) -> Optional[Path]:
        collection_name = getattr(self, "collection_name", None)
        if not self.persist_directory or not collection_name:
            return None
        return Path(self.persist_directory) / f"bm25_{collection_name}.json"

    def _update_bm25(self, documents: List[Document]) -> None:
        texts = [doc.page_content for doc in documents]
        self._bm25.add_documents(texts)
        bm25_path = self._bm25_path()
        if bm25_path:
            self._bm25.save(bm25_path)

    def _bm25_search(self, question: str, k: int) -> List[Document]:
        results = self._bm25.search(question, k)
        docs = []
        for idx, _score in results:
            if idx < len(self._bm25.corpus):
                docs.append(Document(page_content=self._bm25.corpus[idx]))
        return docs

    def _rrf(
        self, ranked_lists: List[List[Document]], k_rrf: int = 60
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:100]
                scores[key] = scores.get(key, 0) + 1 / (k_rrf + rank + 1)
                doc_map[key] = doc
        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def _hybrid_search(
        self,
        question: str,
        k: int,
        retrieval_filter: RetrievalFilter | None,
    ) -> List[Document]:
        fetch_k = k * 2
        semantic_docs = self._semantic_search(question, fetch_k, retrieval_filter)
        bm25_docs = self._bm25_search(question, fetch_k)
        return self._rrf([semantic_docs, bm25_docs])[:k]

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    def _semantic_search(
        self,
        question: str,
        k: int,
        retrieval_filter: RetrievalFilter | None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """Backend-specific dense vector search."""
        raise NotImplementedError

    def add_documents(self, documents: List[Document]) -> None:
        raise NotImplementedError

    def add_class_documents(self, documents: List[Document]) -> None:
        raise NotImplementedError

    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        *,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> List[Document]:
        raise NotImplementedError

    def get_available_collections(self) -> List[str]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared similarity_search with search_type routing
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        question: str,
        k: int = 5,
        *,
        retrieval_filter: RetrievalFilter | None = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        if retrieval_filter is not None:
            retrieval_filter = normalize_retrieval_filter(retrieval_filter)
            if isinstance(retrieval_filter, MatchNone):
                return []
        if self.search_type == "bm25":
            if retrieval_filter is not None:
                raise NotImplementedError(
                    "retrieval_filter is not supported for BM25 search yet"
                )
            return self._bm25_search(question, k)
        elif self.search_type == "hybrid":
            if retrieval_filter is not None:
                raise NotImplementedError(
                    "retrieval_filter is not supported for hybrid search yet"
                )
            return self._hybrid_search(question, k, retrieval_filter)
        return self._semantic_search(question, k, retrieval_filter, collection_name)
