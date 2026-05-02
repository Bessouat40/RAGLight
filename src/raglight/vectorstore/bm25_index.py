from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from rank_bm25 import BM25Okapi


@dataclass
class BM25Document:
    """Represents a document in the BM25 index with text and metadata."""
    text: str
    metadata: Dict[str, Any]


class BM25Index:
    """
    Lightweight BM25 index over a list of text documents with metadata support.
    
    Backward compatible: search() returns (index, score) for existing tests.
    Use search_with_metadata() for full results with text and metadata.
    """

    def __init__(self) -> None:
        self.documents: List[BM25Document] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def corpus(self) -> List[str]:
        """Backward compatible: return list of text contents."""
        return [doc.text for doc in self.documents]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _rebuild(self) -> None:
        if self.documents:
            self._bm25 = BM25Okapi([self._tokenize(doc.text) for doc in self.documents])
        else:
            self._bm25 = None

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the BM25 index.
        
        Args:
            texts: List of document text contents.
            metadatas: Optional list of metadata dictionaries corresponding to each text.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        if len(metadatas) != len(texts):
            metadatas = metadatas + [{}] * (len(texts) - len(metadatas))
        
        for text, metadata in zip(texts, metadatas):
            self.documents.append(BM25Document(text=text, metadata=metadata or {}))
        
        self._rebuild()

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Search the BM25 index (backward compatible).
        
        Args:
            query: The search query.
            k: Number of top results to return.
            
        Returns:
            List of tuples: (index, score) - backward compatible format
        """
        if not self._bm25 or not self.documents:
            return []
        
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        return [(idx, score) for idx, score in indexed[:k]]

    def search_with_metadata(self, query: str, k: int) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """
        Search the BM25 index and return results with full metadata.
        
        Args:
            query: The search query.
            k: Number of top results to return.
            
        Returns:
            List of tuples: (index, score, text, metadata)
        """
        if not self._bm25 or not self.documents:
            return []
        
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed[:k]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append((idx, score, doc.text, doc.metadata))
        
        return results

    def save(self, path: Path) -> None:
        data = [asdict(doc) for doc in self.documents]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.documents = [BM25Document(**item) for item in data]
        self._rebuild()
