from __future__ import annotations
import logging
import uuid
from typing import List, Dict, Optional, Any, cast

from ...contracts import EmbeddingModel
from ...models import Document

try:
    from typing_extensions import override
    import chromadb
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "chromadb"),
        extra="chroma",
        feature="Chroma vector store support",
    )

from ._base import VectorStoreAdapter


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """
    Adapter to make an embedding model compatible with ChromaDB's interface.
    """

    def __init__(self, embeddings_model: EmbeddingModel):
        self.embeddings_model = embeddings_model

    def __call__(self, input: Documents) -> Embeddings:
        if hasattr(self.embeddings_model, "embed_documents"):
            return self.embeddings_model.embed_documents(cast(List[str], input))
        else:
            raise TypeError(
                f"Object {type(self.embeddings_model)} does not implement 'embed_documents'."
            )


class ChromaVectorStore(VectorStoreAdapter):
    """
    Concrete implementation for ChromaDB using the official chromadb library.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingModel,
        persist_directory: str = None,
        host: str = None,
        port: int = None,
        search_type: str = "semantic",
        alpha: float = 0.5,
    ) -> None:
        super().__init__(persist_directory, embeddings_model, search_type, alpha)

        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.collection_name = collection_name

        self.embedding_function = ChromaEmbeddingAdapter(self.embeddings_model)

        if host and port:
            self.client = chromadb.HttpClient(host=host, port=port, ssl=False)
        elif persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            raise ValueError(
                "Invalid configuration: provide host/port OR persist_directory."
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        self.collection_classes = self.client.get_or_create_collection(
            name=f"{collection_name}_classes",
            embedding_function=self.embedding_function,
        )

        bm25_path = self._bm25_path()
        if bm25_path and bm25_path.exists():
            self._bm25.load(bm25_path)
        elif search_type in ("bm25", "hybrid"):
            self._rebuild_bm25_from_chroma()

    def _rebuild_bm25_from_chroma(self) -> None:
        result = self.collection.get()
        texts = result.get("documents") or []
        if texts:
            self._bm25.add_documents(texts)

    @override
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} document chunks to ChromaDB collection '{self.collection.name}'..."
        )

        self._add_docs_to_collection(self.collection, documents)
        self._update_bm25(documents)

        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to ChromaDB collection '{self.collection_classes.name}'..."
        )

        self._add_docs_to_collection(self.collection_classes, documents)

        logging.info("✅ Class documents successfully added to the class collection.")

    def _add_docs_to_collection(
        self, collection: Any, documents: List[Document]
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [
            doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents
        ]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    @override
    def _semantic_search(
        self,
        question: str,
        k: int,
        filter: Optional[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        target_collection = self.collection
        if collection_name and collection_name != self.collection.name:
            target_collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
        return self._query_collection(target_collection, question, k, filter)

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        target_collection = self.collection_classes

        if collection_name:
            class_col_name = f"{collection_name}_classes"
            if class_col_name != self.collection_classes.name:
                target_collection = self.client.get_or_create_collection(
                    name=class_col_name, embedding_function=self.embedding_function
                )

        return self._query_collection(target_collection, question, k, filter)

    def _query_collection(
        self, collection: Any, question: str, k: int, filter: Optional[Dict[str, Any]]
    ) -> List[Document]:
        results = collection.query(query_texts=[question], n_results=k, where=filter)

        found_docs: List[Document] = []
        if results["documents"] and results["documents"][0]:
            docs_list = results["documents"][0]
            metas_list = (
                results["metadatas"][0]
                if results["metadatas"]
                else [{}] * len(docs_list)
            )
            for text, meta in zip(docs_list, metas_list):
                safe_meta = meta if isinstance(meta, dict) else {}
                found_docs.append(Document(page_content=text, metadata=safe_meta))

        return found_docs

    @override
    def get_available_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logging.error(f"Error listing collections: {e}")
            return []
