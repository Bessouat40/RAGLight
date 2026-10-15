"""Unit tests for the intentionally small v4 public API."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from dataclasses import FrozenInstanceError

import raglight
from raglight import PipelineConfig, RAGPipeline


class PublicAPITests(unittest.TestCase):
    def test_root_exports_only_common_v4_entry_points(self):
        self.assertEqual(["PipelineConfig", "RAGPipeline"], raglight.__all__)
        self.assertIs(raglight.PipelineConfig, PipelineConfig)
        self.assertIs(raglight.RAGPipeline, RAGPipeline)

    def test_legacy_implementation_types_are_not_root_exports(self):
        for name in ("Builder", "RAG", "Settings", "OllamaModel", "ChromaVS"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(raglight, name))

    def test_importing_root_does_not_import_provider_or_delivery_modules(self):
        script = textwrap.dedent("""
            import sys
            import raglight

            forbidden = (
                "raglight.api",
                "raglight.cli",
                "raglight.ui",
                "raglight.llm.ollama_model",
                "raglight.llm.openai_model",
                "raglight.vectorstore.chroma",
                "raglight.vectorstore.qdrant",
                "fastapi",
                "langchain",
                "langgraph",
                "ollama",
                "openai",
                "streamlit",
            )
            loaded = [name for name in forbidden if name in sys.modules]
            raise SystemExit(f"unexpected imports: {loaded}" if loaded else 0)
            """)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)


class PipelineConfigTests(unittest.TestCase):
    def test_top_k_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "top_k must be greater than zero"):
            PipelineConfig(top_k=0)

    def test_configuration_is_immutable(self):
        config = PipelineConfig(top_k=3)

        with self.assertRaises(FrozenInstanceError):
            config.top_k = 4


if __name__ == "__main__":
    unittest.main()
