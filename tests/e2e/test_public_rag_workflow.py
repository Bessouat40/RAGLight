import unittest
from uuid import UUID

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import IntegrationRegistries, IntegrationRegistry
from raglight.models import Document


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search(self, question, k=5):
        self.calls.append((question, k))
        return [Document(page_content="RAGLight is a modular RAG library.")]


class FakeLLM:
    def __init__(self):
        self.inputs = []

    def generate(self, input):
        self.inputs.append(input)
        return "RAGLight is modular."

    def generate_streaming(self, input, callbacks=None):
        self.inputs.append(input)
        yield "RAGLight "
        yield "is modular."


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, question, documents, top_k):
        self.calls.append((question, documents, top_k))
        return list(reversed(documents))[:top_k]


class PublicRAGWorkflowTests(unittest.TestCase):
    def test_question_to_answer_workflow_uses_public_import(self):
        embeddings = FakeEmbeddings()
        vector_store = FakeVectorStore()
        llm = FakeLLM()
        pipeline = RAGPipeline(
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            config=PipelineConfig(top_k=1),
        )

        response = pipeline.generate("What is RAGLight?")

        self.assertEqual("RAGLight is modular.", response.answer)
        self.assertEqual(
            "RAGLight is a modular RAG library.", response.sources[0].page_content
        )
        self.assertEqual(
            ["retrieval", "generation"],
            [event.stage for event in response.events],
        )
        self.assertEqual(
            ["completed", "completed"],
            [event.status for event in response.events],
        )
        self.assertEqual({"retrieval", "generation", "total"}, set(response.timings))
        for event in response.events:
            self.assertEqual(response.timings[event.stage], event.duration_ms)
        self.assertGreaterEqual(
            response.timings["total"],
            response.timings["retrieval"] + response.timings["generation"],
        )
        self.assertEqual(
            {"top_k": 1, "retrieved_count": 1, "reranked": False},
            dict(response.metadata),
        )
        self.assertEqual(response.trace_id, str(UUID(response.trace_id)))
        self.assertEqual([("What is RAGLight?", 1)], vector_store.calls)
        self.assertIn("RAGLight is a modular RAG library.", llm.inputs[0]["question"])

    def test_explicit_history_is_forwarded_without_pipeline_state(self):
        llm = FakeLLM()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            llm=llm,
        )
        history = [{"role": "user", "content": "Earlier question"}]

        pipeline.generate("Follow-up question", history=history)

        self.assertEqual(history, llm.inputs[0]["history"])
        self.assertFalse(hasattr(pipeline, "state"))

    def test_streaming_uses_the_same_public_pipeline(self):
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
        )

        chunks = list(pipeline.stream("What is RAGLight?"))

        self.assertEqual(["RAGLight ", "is modular."], chunks)

    def test_session_preserves_its_own_generated_and_streamed_turns(self):
        llm = FakeLLM()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            llm=llm,
        )
        session = pipeline.create_session(session_id="user-1")

        session.generate("First question")
        chunks = list(session.stream("Follow-up question"))

        self.assertEqual(["RAGLight ", "is modular."], chunks)
        self.assertEqual(
            (
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "RAGLight is modular."},
                {"role": "user", "content": "Follow-up question"},
                {"role": "assistant", "content": "RAGLight is modular."},
            ),
            session.history,
        )
        self.assertEqual(list(session.history[:2]), llm.inputs[1]["history"])

    def test_optional_reranker_is_injected_without_builder_changes(self):
        reranker = FakeReranker()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
            reranker=reranker,
            config=PipelineConfig(top_k=1),
        )

        response = pipeline.generate("What is RAGLight?")

        question, documents, top_k = reranker.calls[0]
        self.assertEqual("What is RAGLight?", question)
        self.assertEqual(
            "RAGLight is a modular RAG library.", documents[0].page_content
        )
        self.assertEqual(1, top_k)
        self.assertEqual(
            ["retrieval", "reranking", "generation"],
            [event.stage for event in response.events],
        )
        self.assertTrue(response.metadata["reranked"])

    def test_empty_questions_are_rejected_before_retrieval(self):
        vector_store = FakeVectorStore()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=vector_store,
            llm=FakeLLM(),
        )

        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            pipeline.generate("   ")

        self.assertEqual([], vector_store.calls)

    def test_serialized_names_build_the_public_pipeline(self):
        registries = IntegrationRegistries(
            llms=IntegrationRegistry("LLM"),
            embeddings=IntegrationRegistry("embedding"),
            vector_stores=IntegrationRegistry("vector store"),
            rerankers=IntegrationRegistry("reranker"),
            document_processors=IntegrationRegistry("document processor"),
        )
        registries.llms.register("fake", FakeLLM)
        registries.embeddings.register("fake", FakeEmbeddings)
        registries.vector_stores.register(
            "fake", lambda *, embeddings_model: FakeVectorStore()
        )
        config = PipelineConfig.from_dict(
            {
                "top_k": 1,
                "llm": {"name": "fake"},
                "embeddings": {"name": "fake"},
                "vector_store": {"name": "fake"},
            }
        )

        pipeline = RAGPipeline.from_config(config, registries=registries)

        self.assertEqual("RAGLight is modular.", pipeline.generate("Question").answer)


if __name__ == "__main__":
    unittest.main()
