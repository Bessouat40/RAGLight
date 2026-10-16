"""Integration coverage for the built-in agent planner and retrieval tool."""

from __future__ import annotations

import unittest

from raglight.agentic import AgenticPipeline
from raglight.models import Document


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0]


class InMemoryStore:
    def similarity_search(self, question, k=5):
        return [Document("RAGLight has separate classic and agentic APIs.")]


class PlanningLLM:
    def __init__(self):
        self.calls = []

    def generate(self, input):
        self.calls.append(input)
        if len(self.calls) == 1:
            return '{"type":"tool","tool_name":"retrieve","arguments":{"query":"RAGLight APIs"}}'
        return '{"type":"final","answer":"The APIs are separate."}'

    def generate_streaming(self, input):
        yield self.generate(input)


class AgenticWorkflowIntegrationTests(unittest.TestCase):
    def test_default_planner_retrieves_context_and_finishes(self):
        llm = PlanningLLM()
        pipeline = AgenticPipeline(
            llm=llm,
            embeddings=FakeEmbeddings(),
            vector_store=InMemoryStore(),
            max_steps=3,
        )

        response = pipeline.generate("How are the APIs organized?")

        self.assertEqual("The APIs are separate.", response.answer)
        self.assertEqual(1, len(response.sources))
        self.assertIn(
            "RAGLight has separate", llm.calls[1]["steps"][0]["observation"]["content"]
        )
        self.assertEqual("final", response.metadata["termination"])


if __name__ == "__main__":
    unittest.main()
