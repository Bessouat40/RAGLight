import unittest
from unittest.mock import MagicMock, patch

from raglight.integrations.embeddings.bedrock_embeddings import BedrockEmbeddings
from ..test_config import TestsConfig


class TestBedrockEmbeddings(unittest.TestCase):

    @patch(
        "raglight.integrations.embeddings.bedrock_embeddings.LangChainBedrockEmbeddings"
    )
    def test_model_load(self, mock_bedrock: MagicMock):
        mock_bedrock.return_value = MagicMock()
        model = BedrockEmbeddings(
            model_name=TestsConfig.BEDROCK_EMBEDDING_MODEL,
            region_name="us-east-1",
        )
        self.assertIsNotNone(model.model)
        mock_bedrock.assert_called_once_with(
            model_id=TestsConfig.BEDROCK_EMBEDDING_MODEL,
            region_name="us-east-1",
        )

    @patch(
        "raglight.integrations.embeddings.bedrock_embeddings.LangChainBedrockEmbeddings"
    )
    def test_embed_documents(self, mock_bedrock: MagicMock):
        mock_client = mock_bedrock.return_value
        mock_client.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        model = BedrockEmbeddings(
            model_name=TestsConfig.BEDROCK_EMBEDDING_MODEL,
            region_name="us-east-1",
        )
        result = model.embed_documents(["text1", "text2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2])
        mock_client.embed_documents.assert_called_once_with(["text1", "text2"])

    @patch(
        "raglight.integrations.embeddings.bedrock_embeddings.LangChainBedrockEmbeddings"
    )
    def test_embed_query(self, mock_bedrock: MagicMock):
        mock_client = mock_bedrock.return_value
        mock_client.embed_query.return_value = [0.5, 0.6]

        model = BedrockEmbeddings(
            model_name=TestsConfig.BEDROCK_EMBEDDING_MODEL,
            region_name="us-east-1",
        )
        result = model.embed_query("query text")

        self.assertEqual(result, [0.5, 0.6])
        mock_client.embed_query.assert_called_once_with("query text")


if __name__ == "__main__":
    unittest.main()
