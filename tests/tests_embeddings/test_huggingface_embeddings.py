import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from raglight.embeddings.huggingface_embeddings import HuggingfaceEmbeddingsModel
from ..test_config import TestsConfig


class TestHuggingFaceEmbeddings(unittest.TestCase):

    @patch("raglight.embeddings.huggingface_embeddings.SentenceTransformer")
    def test_model_load(self, mock_transformer: MagicMock):
        """Test SentenceTransformer model loading."""
        mock_transformer.return_value = MagicMock()
        embeddings = HuggingfaceEmbeddingsModel(TestsConfig.HUGGINGFACE_EMBEDDINGS)

        self.assertIsNotNone(embeddings.model, "Model should be loaded successfully.")
        mock_transformer.assert_called_once_with(TestsConfig.HUGGINGFACE_EMBEDDINGS)

    @patch("raglight.embeddings.huggingface_embeddings.SentenceTransformer")
    def test_embed_documents(self, mock_transformer: MagicMock):
        """Test document encoding."""
        mock_instance = mock_transformer.return_value
        mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        embeddings = HuggingfaceEmbeddingsModel(TestsConfig.HUGGINGFACE_EMBEDDINGS)
        result = embeddings.embed_documents(["text1", "text2"])

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        mock_instance.encode.assert_called_with(["text1", "text2"])

    @patch("raglight.embeddings.huggingface_embeddings.SentenceTransformer")
    def test_embed_query(self, mock_transformer: MagicMock):
        """Test query encoding."""
        mock_instance = mock_transformer.return_value
        mock_instance.encode.return_value = np.array([0.5, 0.6])

        embeddings = HuggingfaceEmbeddingsModel(TestsConfig.HUGGINGFACE_EMBEDDINGS)
        result = embeddings.embed_query("query text")

        self.assertIsInstance(result, list)
        self.assertEqual(result, [0.5, 0.6])
        mock_instance.encode.assert_called_with("query text")


if __name__ == "__main__":
    unittest.main()
