from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Optional
import os
import logging
from langchain_core.documents import Document
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..document_processing.document_processor import DocumentProcessor
from ..document_processing.document_processor_factory import DocumentProcessorFactory
from ..embeddings.embeddings_model import EmbeddingsModel
from ..config.settings import Settings
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This class provides a shared ingestion pipeline, BM25/hybrid search logic,
    and defines the abstract methods that concrete implementations must provide.
    """

    def __init__(
        self,
        persist_directory: str,
        embeddings_model: EmbeddingsModel,
        custom_processors: Optional[Dict[str, DocumentProcessor]] = None,
        search_type: str = "semantic",
        alpha: float = 0.5,
    ) -> None:
        self.embeddings_model: EmbeddingsModel = embeddings_model
        self.persist_directory: str = persist_directory
        self.vector_store: Any = None
        self.vector_store_classes: Any = None
        self.custom_processors: Dict[str, DocumentProcessor] = custom_processors or {}
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
        metadatas = [dict(doc.metadata) if doc.metadata else {} for doc in documents]
        self._bm25.add_documents(texts, metadatas)
        bm25_path = self._bm25_path()
        if bm25_path:
            self._bm25.save(bm25_path)

    def _bm25_search(self, question: str, k: int) -> List[Document]:
        results = self._bm25.search_with_metadata(question, k)
        docs = []
        for idx, score, text, metadata in results:
            new_metadata = dict(metadata) if metadata else {}
            new_metadata["bm25_score"] = score
            new_metadata["bm25_index"] = idx
            new_metadata["retrieval_stage"] = "bm25"
            docs.append(Document(page_content=text, metadata=new_metadata))
        logger.info(f"BM25 search: found {len(docs)} documents with metadata")
        return docs

    def _rrf(
        self, ranked_lists: List[List[Document]], k_rrf: int = 60
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:200]
                rrf_score = 1 / (k_rrf + rank + 1)
                
                if key in scores:
                    scores[key] += rrf_score
                    existing_doc = doc_map[key]
                    merged_metadata = dict(existing_doc.metadata)
                    
                    retrieval_stages = merged_metadata.get("retrieval_stages", [])
                    current_stage = doc.metadata.get("retrieval_stage", "unknown")
                    if current_stage not in retrieval_stages:
                        retrieval_stages.append(current_stage)
                    merged_metadata["retrieval_stages"] = retrieval_stages
                    
                    merged_metadata.update(doc.metadata)
                    merged_metadata["rrf_combined_score"] = scores[key]
                    merged_metadata["retrieval_stage"] = "hybrid"
                    doc_map[key] = Document(
                        page_content=existing_doc.page_content,
                        metadata=merged_metadata
                    )
                else:
                    scores[key] = rrf_score
                    new_metadata = dict(doc.metadata) if doc.metadata else {}
                    new_metadata["rrf_score"] = rrf_score
                    
                    retrieval_stage = new_metadata.get("retrieval_stage", "unknown")
                    new_metadata["retrieval_stages"] = [retrieval_stage]
                    new_metadata["retrieval_stage"] = "hybrid"
                    
                    doc_map[key] = Document(
                        page_content=doc.page_content,
                        metadata=new_metadata
                    )
        
        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        result = [doc_map[k] for k in sorted_keys]
        logger.info(f"RRF fusion: merged {len(ranked_lists)} ranked lists into {len(result)} unique documents")
        return result

    def _hybrid_search(
        self, question: str, k: int, filter: Optional[Dict[str, Any]]
    ) -> List[Document]:
        fetch_k = k * 2
        semantic_docs = self._semantic_search(question, fetch_k, filter)
        bm25_docs = self._bm25_search(question, fetch_k)
        return self._rrf([semantic_docs, bm25_docs])[:k]

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _semantic_search(
        self,
        question: str,
        k: int,
        filter: Optional[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """Backend-specific dense vector search."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def add_class_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search_class(
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Document]:
        pass

    @abstractmethod
    def get_available_collections(self) -> List[str]:
        pass

    # ------------------------------------------------------------------
    # Shared similarity_search with search_type routing
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        if self.search_type == "bm25":
            return self._bm25_search(question, k)
        elif self.search_type == "hybrid":
            return self._hybrid_search(question, k, filter)
        return self._semantic_search(question, k, filter, collection_name)

    # ------------------------------------------------------------------
    # Ingestion pipeline (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_file(
        file_path: str, factory: DocumentProcessorFactory, flatten_metadata
    ):
        processor = factory.get_processor(file_path)
        if not processor:
            return [], []
        try:
            processed_docs = processor.process(
                file_path, chunk_size=2500, chunk_overlap=250
            )
            chunks = flatten_metadata(processed_docs.get("chunks", []))
            classes = flatten_metadata(processed_docs.get("classes", []))
            return chunks, classes
        except Exception as e:
            logging.warning(f"⚠️ Error processing {file_path}: {e}")
            return [], []

    def ingest(self, data_path: str, ignore_folders: List[str] = None) -> None:
        if not os.path.isdir(data_path):
            logging.error(f"Provided data_path '{data_path}' is not a valid directory.")
            return

        if ignore_folders is None:
            ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS

        factory = DocumentProcessorFactory(custom_processors=self.custom_processors)

        logging.info(f"⏳ Starting ingestion from '{data_path}'...")

        files_to_process = []
        for root, dirs, files in os.walk(data_path, topdown=True):
            dirs[:] = [
                d
                for d in dirs
                if not self._should_ignore(os.path.join(root, d), ignore_folders)
            ]
            for file in files:
                file_path = os.path.join(root, file)
                processor = factory.get_processor(file_path)
                if processor:
                    logging.info(
                        f"  -> Queuing '{file_path}' with {processor.__class__.__name__}"
                    )
                    files_to_process.append((file_path, processor))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._process_file, file_path, factory, self._flatten_metadata
                )
                for file_path, _ in files_to_process
            ]

            for future in as_completed(futures):
                try:
                    chunks, classes = future.result()
                    if chunks:
                        self.add_documents(chunks)
                    if classes:
                        self.add_class_documents(classes)
                except Exception as e:
                    logging.warning(f"⚠️ Future raised an exception: {e}")

        logging.info("🎉 Ingestion process completed successfully!")

    def _flatten_metadata(self, documents: List[Document]) -> List[Document]:
        cloned_documents = copy.deepcopy(documents)
        for doc in cloned_documents:
            for key, value in doc.metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    doc.metadata[key] = str(value)
        return cloned_documents

    def _should_ignore(self, path: str, ignore_folders: List[str]) -> bool:
        normalized_path = os.path.normpath(path)
        return any(folder in normalized_path.split(os.sep) for folder in ignore_folders)
