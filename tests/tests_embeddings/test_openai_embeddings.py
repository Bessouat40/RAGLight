import unittest
from unittest.mock import MagicMock, patch

from raglight.integrations.embeddings.openai_embeddings import OpenAIEmbeddings
from ..test_config import TestsConfig


class TestOpenAIEmbeddings(unittest.TestCase):

    @patch(
        "raglight.integrations.embeddings.openai_embeddings.LangChainOpenAIEmbeddings"
    )
    def test_model_load(self, mock_embeddings: MagicMock):
        mock_embeddings.return_value = MagicMock()
        model = OpenAIEmbeddings(TestsConfig.OPENAI_EMBEDDING_MODEL)
        self.assertIsNotNone(model.model)
        mock_embeddings.assert_called_once_with(
            model=TestsConfig.OPENAI_EMBEDDING_MODEL,
            api_key=None,
            base_url="https://api.openai.com/v1",
        )

    @patch(
        "raglight.integrations.embeddings.openai_embeddings.LangChainOpenAIEmbeddings"
    )
    def test_embed_documents(self, mock_embeddings: MagicMock):
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_documents.return_value = [[0.1, 0.1], [0.2, 0.2]]

        model = OpenAIEmbeddings(TestsConfig.OPENAI_EMBEDDING_MODEL)
        result = model.embed_documents(["text1", "text2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.1])
        mock_instance.embed_documents.assert_called_with(["text1", "text2"])

    @patch(
        "raglight.integrations.embeddings.openai_embeddings.LangChainOpenAIEmbeddings"
    )
    def test_embed_query(self, mock_embeddings: MagicMock):
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_query.return_value = [0.5, 0.5]

        model = OpenAIEmbeddings(TestsConfig.OPENAI_EMBEDDING_MODEL)
        result = model.embed_query("query")

        self.assertEqual(result, [0.5, 0.5])
        mock_instance.embed_query.assert_called_with("query")


if __name__ == "__main__":
    unittest.main()
