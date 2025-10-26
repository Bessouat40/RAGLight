import unittest
import logging
from unittest.mock import patch, MagicMock
from raglight.config.settings import Settings
from raglight.rag.builder import Builder

logging.getLogger().setLevel(logging.WARNING)

class TestRAGBuilder(unittest.TestCase):
    @patch("raglight.rag.builder.ChromaVS")
    def test_builder_rag(self, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        rag = (
            Builder()
            .with_embeddings(Settings.OLLAMA, model_name="gpt-oss:20b")
            .with_vector_store(
                Settings.CHROMA, persist_directory="/", collection_name="test"
            )
            .with_llm(Settings.OLLAMA, model_name="gpt-oss:20b")
            .build_rag()
        )

        mock_chroma.assert_called_once()
        self.assertIsNotNone(rag)

class TestRATBuilder(unittest.TestCase):
    @patch("raglight.rag.builder.ChromaVS")
    def test_builder_rat(self, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        rat = (
            Builder()
            .with_embeddings(Settings.OLLAMA, model_name="gpt-oss:20b")
            .with_vector_store(
                Settings.CHROMA, persist_directory="/", collection_name="test"
            )
            .with_llm(Settings.OLLAMA, model_name="gpt-oss:20b")
            .with_reasoning_llm(Settings.OLLAMA, model_name="deepseek:r1:20b")
            .build_rat()
        )

        mock_chroma.assert_called_once()
        self.assertIsNotNone(rat)


if __name__ == "__main__":
    unittest.main()
