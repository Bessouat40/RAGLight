import unittest
from unittest.mock import MagicMock, patch

from raglight.integrations.embeddings.ollama_embeddings import OllamaEmbeddings
from ..test_config import TestsConfig


class TestOllamaEmbeddings(unittest.TestCase):

    @patch(
        "raglight.integrations.embeddings.ollama_embeddings.LangChainOllamaEmbeddings"
    )
    def test_model_load(self, mock_embeddings: MagicMock):
        mock_embeddings.return_value = MagicMock()
        embeddings = OllamaEmbeddings(TestsConfig.OLLAMA_EMBEDDING_MODEL)
        self.assertIsNotNone(embeddings.model)
        mock_embeddings.assert_called_once_with(
            model=TestsConfig.OLLAMA_EMBEDDING_MODEL,
            base_url="http://localhost:11434",
        )

    @patch(
        "raglight.integrations.embeddings.ollama_embeddings.LangChainOllamaEmbeddings"
    )
    def test_embed_documents(self, mock_embeddings: MagicMock):
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        model = OllamaEmbeddings(TestsConfig.OLLAMA_EMBEDDING_MODEL)
        texts = ["doc1", "doc2"]
        result = model.embed_documents(texts)

        self.assertEqual(len(result), 2)
        mock_instance.embed_documents.assert_called_with(texts)

    @patch(
        "raglight.integrations.embeddings.ollama_embeddings.LangChainOllamaEmbeddings"
    )
    def test_embed_query(self, mock_embeddings: MagicMock):
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_query.return_value = [0.9, 0.9]

        model = OllamaEmbeddings(TestsConfig.OLLAMA_EMBEDDING_MODEL)
        result = model.embed_query("query text")

        self.assertEqual(result, [0.9, 0.9])
        mock_instance.embed_query.assert_called_with("query text")


if __name__ == "__main__":
    unittest.main()
