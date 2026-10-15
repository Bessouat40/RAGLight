import unittest
from unittest.mock import MagicMock, patch

from raglight.integrations.embeddings.gemini_embeddings import GeminiEmbeddings
from ..test_config import TestsConfig


class TestGeminiEmbeddings(unittest.TestCase):

    @patch(
        "raglight.integrations.embeddings.gemini_embeddings.GoogleGenerativeAIEmbeddings"
    )
    def test_model_load(self, MockEmbeddings: MagicMock):
        model = GeminiEmbeddings(TestsConfig.GEMINI_EMBEDDING_MODEL)
        self.assertTrue(MockEmbeddings.called)
        self.assertIsNotNone(model.model)

    @patch(
        "raglight.integrations.embeddings.gemini_embeddings.GoogleGenerativeAIEmbeddings"
    )
    def test_embed_documents(self, MockEmbeddings: MagicMock):
        mock_instance = MockEmbeddings.return_value
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        model = GeminiEmbeddings(TestsConfig.GEMINI_EMBEDDING_MODEL)
        texts = ["doc1", "doc2"]
        result = model.embed_documents(texts)

        self.assertEqual(len(result), 2)
        mock_instance.embed_documents.assert_called_with(texts)

    @patch(
        "raglight.integrations.embeddings.gemini_embeddings.GoogleGenerativeAIEmbeddings"
    )
    def test_embed_query(self, MockEmbeddings: MagicMock):
        mock_instance = MockEmbeddings.return_value
        mock_instance.embed_query.return_value = [0.1, 0.2]

        model = GeminiEmbeddings(TestsConfig.GEMINI_EMBEDDING_MODEL)
        result = model.embed_query("query")

        self.assertEqual(len(result), 2)
        mock_instance.embed_query.assert_called_with("query")


if __name__ == "__main__":
    unittest.main()
