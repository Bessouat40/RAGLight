import unittest
from unittest.mock import MagicMock, patch

from raglight.vectorstore.qdrant import QdrantVS
from ..test_config import TestsConfig


class TestQdrantVectorStore(unittest.TestCase):

    @patch("qdrant_client.QdrantClient")
    def test_initialization_with_host(self, MockQdrantClient):
        """Test QdrantVS initialization with host parameter."""
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        store = QdrantVS(
            embeddings_model=embeddings,
            collection_name=TestsConfig.COLLECTION_NAME,
            host="localhost",
            port=6333,
        )

        MockQdrantClient.assert_called_with(host="localhost", port=6333)
        self.assertIsNotNone(store.client)

    @patch("qdrant_client.QdrantClient")
    def test_initialization_with_persist_directory(self, MockQdrantClient):
        """Test QdrantVS initialization with persist_directory."""
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        store = QdrantVS(
            embeddings_model=embeddings,
            collection_name=TestsConfig.COLLECTION_NAME,
            persist_directory="/tmp/qdrant",
        )

        MockQdrantClient.assert_called_with(path="/tmp/qdrant")
        self.assertIsNotNone(store.client)

    @patch("qdrant_client.QdrantClient")
    def test_add_documents(self, MockQdrantClient):
        """Test that add_documents processes documents and calls embeddings."""
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        store = QdrantVS(
            embeddings_model=embeddings,
            collection_name=TestsConfig.COLLECTION_NAME,
            host="localhost",
        )

        from langchain_core.documents import Document

        documents = [Document(page_content="test", metadata={"source": "test.txt"})]
        store.add_documents(documents)

        embeddings.embed_documents.assert_called_once()


if __name__ == "__main__":
    unittest.main()
