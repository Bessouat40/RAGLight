"""Unit tests for the legacy cross-encoder compatibility adapter."""

from __future__ import annotations

import unittest

from raglight.cross_encoder.cross_encoder_model import CrossEncoderModel
from raglight.models import Document


class FakeCrossEncoder(CrossEncoderModel):
    def load(self):
        return object()

    def predict(self, query, documents, top_k):
        return [documents[1], documents[0]][:top_k]


class RerankerAdapterTests(unittest.TestCase):
    def test_rerank_preserves_documents_and_metadata_in_ranked_order(self):
        first = Document(page_content="first", metadata={"source": "a"})
        second = Document(page_content="second", metadata={"source": "b"})

        result = FakeCrossEncoder("fake").rerank("question", [first, second], top_k=2)

        self.assertEqual([second, first], result)
        self.assertEqual({"source": "b"}, result[0].metadata)


if __name__ == "__main__":
    unittest.main()
