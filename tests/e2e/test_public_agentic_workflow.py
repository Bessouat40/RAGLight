"""End-to-end tests through the separate public agentic namespace."""

from __future__ import annotations

import unittest

from raglight.agentic import AgentDecision, AgenticPipeline, FunctionTool


class UnusedLLM:
    def generate(self, input):
        raise AssertionError("a custom agent should own decisions")

    def generate_streaming(self, input):
        raise AssertionError("agentic streaming is not part of this workflow")


class Embeddings:
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0]


class Store:
    def similarity_search(self, question, k=5):
        return []


class PublicCustomAgent:
    def decide(self, context):
        if not context.steps:
            return AgentDecision.call_tool("uppercase", {"text": context.question})
        observation = context.steps[-1].observation
        assert observation is not None
        return AgentDecision.final(observation.content)


class PublicAgenticWorkflowTests(unittest.TestCase):
    def test_custom_agent_and_tool_complete_a_public_workflow(self):
        pipeline = AgenticPipeline(
            llm=UnusedLLM(),
            embeddings=Embeddings(),
            vector_store=Store(),
            agent=PublicCustomAgent(),
            tools=[
                FunctionTool(
                    "uppercase",
                    "Convert text to uppercase.",
                    lambda arguments: arguments["text"].upper(),
                )
            ],
        )

        response = pipeline.generate("agentic rag")

        self.assertEqual("AGENTIC RAG", response.answer)
        self.assertEqual("final", response.metadata["termination"])
        self.assertEqual([], list(response.sources))


if __name__ == "__main__":
    unittest.main()
