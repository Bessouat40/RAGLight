"""Integration tests for registry-backed pipeline construction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import IntegrationRegistries, IntegrationRegistry
from raglight.models import Document


class ConfiguredLLM:
    def __init__(self, *, prefix: str) -> None:
        self.prefix = prefix

    def generate(self, input):
        return f"{self.prefix}{input['question']}"

    def generate_streaming(self, input):
        yield self.generate(input)


class ConfiguredEmbeddings:
    def __init__(self, *, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts):
        return [[0.0] * self.dimensions for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dimensions


class ConfiguredVectorStore:
    def __init__(self, *, embeddings_model, content: str) -> None:
        self.embeddings_model = embeddings_model
        self.content = content

    def similarity_search(self, question, k=5):
        return [Document(page_content=self.content)][:k]


def custom_registries() -> IntegrationRegistries:
    registries = IntegrationRegistries(
        llms=IntegrationRegistry("LLM"),
        embeddings=IntegrationRegistry("embedding"),
        vector_stores=IntegrationRegistry("vector store"),
        rerankers=IntegrationRegistry("reranker"),
        document_processors=IntegrationRegistry("document processor"),
    )
    registries.llms.register("custom", ConfiguredLLM)
    registries.embeddings.register("custom", ConfiguredEmbeddings)
    registries.vector_stores.register("custom", ConfiguredVectorStore)
    return registries


class RegistryConfigurationTests(unittest.TestCase):
    def test_pipeline_is_built_from_serialized_custom_components(self):
        config = PipelineConfig.from_dict(
            {
                "top_k": 2,
                "llm": {"name": "custom", "options": {"prefix": "Answer: "}},
                "embeddings": {
                    "name": "custom",
                    "options": {"dimensions": 3},
                },
                "vector_store": {
                    "name": "custom",
                    "options": {"content": "configured context"},
                },
            }
        )

        pipeline = RAGPipeline.from_config(config, registries=custom_registries())
        answer = pipeline.generate("What is configured?")

        self.assertIn("Answer: ", answer)
        self.assertIn("configured context", answer)

    def test_json_configuration_file_is_supported(self):
        payload = {
            "top_k": 1,
            "llm": {"name": "custom", "options": {"prefix": "A: "}},
            "embeddings": {"name": "custom", "options": {"dimensions": 2}},
            "vector_store": {
                "name": "custom",
                "options": {"content": "from JSON"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raglight.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            config = PipelineConfig.from_file(path)
            pipeline = RAGPipeline.from_config(config, registries=custom_registries())

        self.assertIn("from JSON", pipeline.generate("Question"))

    def test_toml_configuration_file_is_supported(self):
        content = """
top_k = 1

[llm]
name = "custom"
[llm.options]
prefix = "A: "

[embeddings]
name = "custom"
[embeddings.options]
dimensions = 2

[vector_store]
name = "custom"
[vector_store.options]
content = "from TOML"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raglight.toml"
            path.write_text(content, encoding="utf-8")

            pipeline = RAGPipeline.from_config(
                PipelineConfig.from_file(path), registries=custom_registries()
            )

        self.assertIn("from TOML", pipeline.generate("Question"))

    def test_named_construction_requires_core_component_specs(self):
        config = PipelineConfig()

        with self.assertRaisesRegex(ValueError, "llm, embeddings, vector_store"):
            RAGPipeline.from_config(config, registries=custom_registries())


if __name__ == "__main__":
    unittest.main()
