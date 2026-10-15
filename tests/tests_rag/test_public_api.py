"""Unit tests for the intentionally small v4 public API."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from dataclasses import FrozenInstanceError

import raglight
from raglight import PipelineConfig, RAGPipeline
from raglight.models import ComponentConfig, Document
from raglight.pipeline import ConversationSession


class PublicAPITests(unittest.TestCase):
    def test_root_exports_only_common_v4_entry_points(self):
        self.assertEqual(["PipelineConfig", "RAGPipeline"], raglight.__all__)
        self.assertIs(raglight.PipelineConfig, PipelineConfig)
        self.assertIs(raglight.RAGPipeline, RAGPipeline)

    def test_pipeline_namespace_exports_the_session_type(self):
        import raglight.pipeline

        self.assertEqual(
            ["ConversationSession", "PipelineConfig", "RAGPipeline"],
            raglight.pipeline.__all__,
        )
        self.assertIs(raglight.pipeline.ConversationSession, ConversationSession)

    def test_legacy_implementation_types_are_not_root_exports(self):
        for name in ("Builder", "RAG", "Settings", "OllamaModel", "ChromaVS"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(raglight, name))

    def test_importing_root_does_not_import_provider_or_delivery_modules(self):
        script = textwrap.dedent("""
            import sys
            import raglight

            forbidden = (
                "raglight.integrations.ollama",
                "raglight.integrations.openai",
                "raglight.integrations.chroma",
                "raglight.integrations.qdrant",
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

    def test_removed_v3_packages_are_not_importable(self):
        for module in ("raglight.rag", "raglight.config", "raglight.api"):
            with self.subTest(module=module):
                self.assertIsNone(__import__("importlib.util").util.find_spec(module))


class PipelineConfigTests(unittest.TestCase):
    def test_top_k_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "top_k must be greater than zero"):
            PipelineConfig(top_k=0)

    def test_configuration_is_immutable(self):
        config = PipelineConfig(top_k=3)

        with self.assertRaises(FrozenInstanceError):
            config.top_k = 4

    def test_serialized_component_names_are_normalized(self):
        config = PipelineConfig.from_dict(
            {"llm": {"name": " OpenAI ", "options": {"model_name": "model"}}}
        )

        self.assertEqual(
            ComponentConfig(name="openai", options={"model_name": "model"}),
            config.llm,
        )

    def test_unknown_serialized_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown pipeline.*unexpected"):
            PipelineConfig.from_dict({"unexpected": True})

    def test_component_options_are_immutable(self):
        component = ComponentConfig(name="custom", options={"value": 1})

        with self.assertRaises(TypeError):
            component.options["value"] = 2


class DocumentTests(unittest.TestCase):
    def test_metadata_defaults_are_isolated_between_documents(self):
        first = Document(page_content="first")
        second = Document(page_content="second")

        first.metadata["source"] = "one"

        self.assertEqual({}, second.metadata)


if __name__ == "__main__":
    unittest.main()
