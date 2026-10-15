"""Unit tests for the Hugging Face reranker integration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from raglight.models import Document


class HuggingFaceRerankerTests(unittest.TestCase):
    @patch("raglight.integrations.rerankers.huggingface.CrossEncoder")
    def test_rerank_returns_original_documents_in_ranked_order(
        self, cross_encoder: MagicMock
    ) -> None:
        cross_encoder.return_value.rank.return_value = [
            {"corpus_id": 1, "score": 0.9},
            {"corpus_id": 0, "score": 0.5},
        ]
        from raglight.integrations.huggingface import HuggingFaceReranker

        first = Document(page_content="first", metadata={"source": "a"})
        second = Document(page_content="second", metadata={"source": "b"})

        result = HuggingFaceReranker("fake").rerank(
            "question", [first, second], top_k=2
        )

        self.assertEqual([second, first], result)


if __name__ == "__main__":
    unittest.main()
