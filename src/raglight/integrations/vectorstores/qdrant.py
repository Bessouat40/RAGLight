from __future__ import annotations
import logging
import uuid
from typing import List, Dict, Optional, Any
from typing_extensions import override

from ...contracts import EmbeddingModel
from ...models import Document
from ._base import VectorStoreAdapter


class QdrantVectorStore(VectorStoreAdapter):
    """
    Concrete implementation for Qdrant using the qdrant-client library.

    Supports local (on-disk) and remote (HTTP) modes.
    Supports search_type: "semantic" (default), "bm25", "hybrid".
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingModel,
        persist_directory: str = None,
        host: str = None,
        port: int = 6333,
        search_type: str = "semantic",
        alpha: float = 0.5,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ModuleNotFoundError as error:
            from ..._optional import optional_import_error

            optional_import_error(
                error,
                dependency="qdrant_client",
                extra="qdrant",
                feature="Qdrant vector store support",
            )

        super().__init__(persist_directory, embeddings_model, search_type, alpha)

        self.collection_name = collection_name
        self._classes_collection_name = f"{collection_name}_classes"

        if host:
            self.client = QdrantClient(host=host, port=port)
        elif persist_directory:
            self.client = QdrantClient(path=persist_directory)
        else:
            raise ValueError(
                "Invalid configuration: provide host OR persist_directory."
            )

        # Probe vector dimension using a dummy embedding
        sample = embeddings_model.embed_query("probe")
        self._vector_size = len(sample)

        self._ensure_collection(self.collection_name)
        self._ensure_collection(self._classes_collection_name)

        bm25_path = self._bm25_path()
        if bm25_path and bm25_path.exists():
            self._bm25.load(bm25_path)
        elif search_type in ("bm25", "hybrid"):
            self._rebuild_bm25_from_qdrant()

    def _rebuild_bm25_from_qdrant(self) -> None:
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name, limit=10_000, with_payload=True
            )
            texts = [
                r.payload.get("page_content", "")
                for r in records
                if r.payload and r.payload.get("page_content")
            ]
            if texts:
                self._bm25.add_documents(texts)
        except Exception as e:
            logging.warning(f"Could not rebuild BM25 from Qdrant: {e}")

    def _ensure_collection(self, name: str) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def _add_to_collection(
        self, collection_name: str, documents: List[Document]
    ) -> None:
        from qdrant_client.models import PointStruct

        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings_model.embed_documents(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": text,
                    **(doc.metadata if isinstance(doc.metadata, dict) else {}),
                },
            )
            for text, vector, doc in zip(texts, vectors, documents)
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    @override
    def _semantic_search(
        self,
        question: str,
        k: int,
        filter: Optional[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        target = collection_name or self.collection_name
        query_vector = self.embeddings_model.embed_query(question)

        qdrant_filter = None
        if filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filter.items()
                ]
            )

        results = self.client.query_points(
            collection_name=target,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
        ).points

        docs = []
        for hit in results:
            payload = hit.payload or {}
            page_content = payload.pop("page_content", "")
            docs.append(Document(page_content=page_content, metadata=payload))
        return docs

    @override
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        logging.info(
            f"⏳ Adding {len(documents)} document chunks to Qdrant collection '{self.collection_name}'..."
        )
        self._add_to_collection(self.collection_name, documents)
        self._update_bm25(documents)
        logging.info("✅ Documents successfully added.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        logging.info(
            f"⏳ Adding {len(documents)} class documents to Qdrant collection '{self._classes_collection_name}'..."
        )
        self._add_to_collection(self._classes_collection_name, documents)
        logging.info("✅ Class documents successfully added.")

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        if collection_name:
            target = f"{collection_name}_classes"
        else:
            target = self._classes_collection_name
        return self._semantic_search(question, k, filter, target)

    @override
    def get_available_collections(self) -> List[str]:
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            logging.error(f"Error listing Qdrant collections: {e}")
            return []
