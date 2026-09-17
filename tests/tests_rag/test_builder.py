import unittest
import logging
from unittest.mock import patch, MagicMock
from raglight.config.settings import Settings
from raglight.rag.builder import Builder

logging.getLogger().setLevel(logging.WARNING)


class TestRAGBuilder(unittest.TestCase):
    @patch("raglight.vectorstore.chroma.ChromaVS")
    def test_builder_rag(self, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        rag = (
            Builder()
            .with_embeddings(Settings.OLLAMA, model_name="gpt-oss:20b")
            .with_vector_store(
                Settings.CHROMA, persist_directory="/", collection_name="test"
            )
            .with_llm(Settings.OLLAMA, model_name="gpt-oss:20b", preload_model=False)
            .build_rag()
        )

        mock_chroma.assert_called_once()
        self.assertIsNotNone(rag)

    @patch("raglight.vectorstore.chroma.ChromaVS")
    def test_builder_with_huggingface_embeddings(self, mock_chroma):
        """Test builder with HuggingFace embeddings."""
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        rag = (
            Builder()
            .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
            .with_vector_store(
                Settings.CHROMA, persist_directory="/", collection_name="test"
            )
            .with_llm(Settings.OLLAMA, model_name="llama3.2:1b", preload_model=False)
            .build_rag()
        )

        self.assertIsNotNone(rag)
        self.assertIsNotNone(rag.llm)

    @patch("raglight.vectorstore.qdrant.QdrantVS")
    def test_builder_with_qdrant(self, mock_qdrant):
        """Test builder with Qdrant vector store."""
        mock_qdrant_instance = MagicMock()
        mock_qdrant.return_value = mock_qdrant_instance

        rag = (
            Builder()
            .with_embeddings(Settings.OLLAMA, model_name="nomic-embed-text")
            .with_vector_store(
                Settings.QDRANT, location=":memory:", collection_name="test"
            )
            .with_llm(Settings.OLLAMA, model_name="llama3.2:1b", preload_model=False)
            .build_rag()
        )

        mock_qdrant.assert_called_once()
        self.assertIsNotNone(rag)

    @patch("raglight.vectorstore.chroma.ChromaVS")
    @patch("raglight.Settings.OPENAI_API_KEY", "DUMMY_KEY")
    def test_builder_with_openai_embeddings(self, mock_chroma):
        """Test builder with OpenAI embeddings."""
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        rag = (
            Builder()
            .with_embeddings(Settings.OLLAMA, model_name="nomic-embed-text")
            .with_vector_store(
                Settings.CHROMA, persist_directory="/", collection_name="test"
            )
            .with_llm(Settings.OLLAMA, model_name="llama3.2:1b", preload_model=False)
            .build_rag()
        )

        self.assertIsNotNone(rag)
        self.assertIsNotNone(rag.llm)


if __name__ == "__main__":
    unittest.main()
