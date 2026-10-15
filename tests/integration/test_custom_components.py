"""Integration coverage for dependency-injected extension components."""

from __future__ import annotations

import unittest

from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document


class CustomLLM:
    def generate(self, input):
        return input["question"]

    def generate_streaming(self, input, callbacks=None):
        yield self.generate(input)


class CustomEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


class CustomVectorStore:
    def similarity_search(self, question, k=5):
        return [
            Document(page_content="lower-ranked"),
            Document(page_content="higher-ranked"),
        ][:k]


class CustomReranker:
    def rerank(self, question, documents, top_k):
        return list(reversed(documents))[:top_k]


class CustomComponentIntegrationTests(unittest.TestCase):
    def test_complete_custom_component_set_runs_without_builder_registration(self):
        pipeline = RAGPipeline(
            llm=CustomLLM(),
            embeddings=CustomEmbeddings(),
            vector_store=CustomVectorStore(),
            reranker=CustomReranker(),
            config=PipelineConfig(top_k=2),
        )

        answer = pipeline.generate("Which document ranks first?")

        self.assertLess(answer.index("higher-ranked"), answer.index("lower-ranked"))


if __name__ == "__main__":
    unittest.main()
