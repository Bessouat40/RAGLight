from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi


class BM25Index:
    """Lightweight BM25 index over a list of text documents."""

    def __init__(self) -> None:
        self.corpus: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _rebuild(self) -> None:
        if self.corpus:
            self._bm25 = BM25Okapi([self._tokenize(t) for t in self.corpus])
        else:
            self._bm25 = None

    def add_documents(self, texts: List[str]) -> None:
        self.corpus.extend(texts)
        self._rebuild()

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self._bm25 or not self.corpus:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in indexed[:k]]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.corpus, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.corpus = json.loads(path.read_text(encoding="utf-8"))
        self._rebuild()
