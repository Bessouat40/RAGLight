"""Integration coverage for dependency-injected extension components."""

from __future__ import annotations

import unittest

from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document


class CustomLLM:
    def __init__(self):
        self.inputs = []

    def generate(self, input):
        self.inputs.append(input)
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

        response = pipeline.generate("Which document ranks first?")

        self.assertLess(
            response.answer.index("higher-ranked"),
            response.answer.index("lower-ranked"),
        )
        self.assertEqual(
            ["higher-ranked", "lower-ranked"],
            [source.page_content for source in response.sources],
        )

    def test_sessions_share_components_without_sharing_conversation_state(self):
        llm = CustomLLM()
        pipeline = RAGPipeline(
            llm=llm,
            embeddings=CustomEmbeddings(),
            vector_store=CustomVectorStore(),
        )
        first = pipeline.create_session(session_id="first")
        second = pipeline.create_session(session_id="second")

        first.generate("First conversation")
        second.generate("Second conversation")
        first.generate("First follow-up")

        self.assertEqual([], llm.inputs[0]["history"])
        self.assertEqual([], llm.inputs[1]["history"])
        self.assertEqual(
            [
                {"role": "user", "content": "First conversation"},
                {"role": "assistant", "content": llm.inputs[0]["question"]},
            ],
            llm.inputs[2]["history"],
        )


if __name__ == "__main__":
    unittest.main()
