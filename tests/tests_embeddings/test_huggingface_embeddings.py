import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from raglight.integrations.embeddings.huggingface_embeddings import (
    HuggingFaceEmbeddings,
)
from ..test_config import TestsConfig


class TestHuggingFaceEmbeddings(unittest.TestCase):

    @patch(
        "raglight.integrations.embeddings.huggingface_embeddings.SentenceTransformer"
    )
    def test_model_load(self, mock_transformer: MagicMock):
        """Test SentenceTransformer model loading."""
        mock_transformer.return_value = MagicMock()
        embeddings = HuggingFaceEmbeddings(TestsConfig.HUGGINGFACE_EMBEDDINGS)

        self.assertIsNotNone(embeddings.model, "Model should be loaded successfully.")
        mock_transformer.assert_called_once_with(TestsConfig.HUGGINGFACE_EMBEDDINGS)

    @patch(
        "raglight.integrations.embeddings.huggingface_embeddings.SentenceTransformer"
    )
    def test_embed_documents(self, mock_transformer: MagicMock):
        """Test document encoding."""
        mock_instance = mock_transformer.return_value
        mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        embeddings = HuggingFaceEmbeddings(TestsConfig.HUGGINGFACE_EMBEDDINGS)
        result = embeddings.embed_documents(["text1", "text2"])

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        mock_instance.encode.assert_called_with(["text1", "text2"])


if __name__ == "__main__":
    unittest.main()
