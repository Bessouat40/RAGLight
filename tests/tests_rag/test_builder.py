import subprocess
import sys
import textwrap
import unittest
import logging
from unittest.mock import patch, MagicMock
from raglight.config.settings import Settings
from raglight.rag.builder import Builder

logging.getLogger().setLevel(logging.WARNING)


class TestRAGBuilder(unittest.TestCase):
    def test_import_does_not_load_concrete_provider_integrations(self):
        script = textwrap.dedent("""
            import sys
            from raglight.rag.builder import Builder

            forbidden = (
                "langchain_aws",
                "langchain_google_genai",
                "langchain_mistralai",
                "langchain_ollama",
                "langchain_openai",
                "sentence_transformers",
            )
            loaded = [
                name for name in forbidden
                if name in sys.modules
                or any(module.startswith(f"{name}.") for module in sys.modules)
            ]
            raise SystemExit(f"unexpected imports: {loaded}" if loaded else 0)
            """)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)

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


if __name__ == "__main__":
    unittest.main()
