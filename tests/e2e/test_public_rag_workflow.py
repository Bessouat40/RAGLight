import unittest
from contextlib import contextmanager
from uuid import UUID

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import (
    IntegrationRegistries,
    IntegrationRegistry,
    default_registries,
)
from raglight.models import And, Document, Eq, FolderSource, In, MatchNone


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


class ScopedVectorStore:
    def __init__(self):
        self.calls = []
        self.documents = [
            Document("Alice architecture", {"owner_id": "alice", "file_id": "a"}),
            Document("Bob architecture", {"owner_id": "bob", "file_id": "b"}),
        ]

    def similarity_search(self, question, k=5, *, retrieval_filter=None):
        self.calls.append((question, k, retrieval_filter))
        if isinstance(retrieval_filter, Eq):
            documents = [
                document
                for document in self.documents
                if document.metadata.get(retrieval_filter.field)
                == retrieval_filter.value
            ]
        else:
            documents = self.documents
        return documents[:k]


class InMemoryIngestionStore:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def similarity_search(self, question, k=5):
        return self.documents[:k]


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


class ReformulatingLLM(FakeLLM):
    def generate(self, input):
        self.inputs.append(input)
        if "Rewrite the follow-up question" in input["question"]:
            return "Which vector store does this project use?"
        return "The project uses Qdrant."


class RecordedObservation:
    def __init__(self, record):
        self.record = record

    def update(self, **values):
        self.record.setdefault("updates", []).append(values)


class RecordingTracer:
    def __init__(self):
        self.records = []

    @contextmanager
    def observe(self, **values):
        record = {**values, "updates": []}
        self.records.append(record)
        yield RecordedObservation(record)


class PublicRAGWorkflowTests(unittest.TestCase):
    def test_scoped_retrieval_returns_only_documents_in_the_requested_scope(self):
        store = ScopedVectorStore()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=store,
            llm=FakeLLM(),
        )

        alice = pipeline.generate(
            "Explain the architecture",
            retrieval_filter=Eq("owner_id", "alice"),
        )
        bob = pipeline.generate(
            "Explain the architecture",
            retrieval_filter=Eq("owner_id", "bob"),
        )

        self.assertEqual(
            ["Alice architecture"], [doc.page_content for doc in alice.sources]
        )
        self.assertEqual(
            ["Bob architecture"], [doc.page_content for doc in bob.sources]
        )

    def test_reformulation_and_streaming_preserve_the_same_scope(self):
        scope = And(Eq("workspace_id", "ws-1"), In("file_id", ["file-1"]))
        llm = ReformulatingLLM()
        store = ScopedVectorStore()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=store,
            llm=llm,
            config=PipelineConfig(reformulate_queries=True),
        )

        pipeline.generate(
            "What does it use?",
            history=[{"role": "user", "content": "Earlier"}],
            retrieval_filter=scope,
        )
        list(pipeline.stream("Stream it", retrieval_filter=scope))

        self.assertEqual(scope, store.calls[0][2])
        self.assertEqual(scope, store.calls[1][2])

    def test_match_none_never_queries_the_vector_store(self):
        store = ScopedVectorStore()
        llm = FakeLLM()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=store,
            llm=llm,
        )

        response = pipeline.generate("Question", retrieval_filter=MatchNone())

        self.assertEqual([], store.calls)
        self.assertEqual((), response.sources)

    def test_scoped_retrieval_does_not_silently_use_a_legacy_adapter(self):
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
        )

        with self.assertRaisesRegex(TypeError, "retrieval_filter"):
            pipeline.generate("Question", retrieval_filter=Eq("owner_id", "alice"))

    def test_nested_file_is_ingested_retrieved_and_used_for_generation(self):
        from pathlib import Path
        import tempfile

        store = InMemoryIngestionStore()
        llm = FakeLLM()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=store,
            llm=llm,
            document_processors=default_registries.document_processors,
        )

        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "nested"
            nested.mkdir()
            path = nested / "knowledge.txt"
            path.write_text(
                "Directory ingestion reaches the public RAG workflow.",
                encoding="utf-8",
            )

            ingestion = pipeline.ingest([FolderSource(path=directory)])
            response = pipeline.generate("What reaches the workflow?")

        self.assertEqual(1, ingestion.processed_count)
        self.assertEqual(1, ingestion.indexed_count)
        self.assertEqual("RAGLight is modular.", response.answer)
        self.assertIn("Directory ingestion reaches", llm.inputs[0]["question"])

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

    def test_reformulation_uses_history_for_retrieval_and_preserves_original_question(
        self,
    ):
        llm = ReformulatingLLM()
        vector_store = FakeVectorStore()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=vector_store,
            llm=llm,
            config=PipelineConfig(reformulate_queries=True),
        )
        history = [{"role": "user", "content": "This project uses Qdrant."}]

        response = pipeline.generate("What does it use?", history=history)

        self.assertEqual(
            [("Which vector store does this project use?", 5)],
            vector_store.calls,
        )
        self.assertIn("Question:\nWhat does it use?", llm.inputs[1]["question"])
        self.assertEqual(history, llm.inputs[1]["history"])
        self.assertEqual(
            ["reformulation", "retrieval", "generation"],
            [event.stage for event in response.events],
        )
        self.assertEqual(
            "Which vector store does this project use?",
            response.metadata["retrieval_query"],
        )

    def test_reformulation_skips_the_extra_call_without_history(self):
        llm = ReformulatingLLM()
        vector_store = FakeVectorStore()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=vector_store,
            llm=llm,
            config=PipelineConfig(reformulate_queries=True),
        )

        response = pipeline.generate("Standalone question")

        self.assertEqual([("Standalone question", 5)], vector_store.calls)
        self.assertEqual(1, len(llm.inputs))
        self.assertNotIn("retrieval_query", response.metadata)

    def test_tracer_receives_root_and_stage_observations(self):
        tracer = RecordingTracer()
        pipeline = RAGPipeline(
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
            tracer=tracer,
        )

        response = pipeline.generate("What is RAGLight?")

        self.assertEqual(
            ["raglight.rag", "raglight.retrieval", "raglight.generation"],
            [record["name"] for record in tracer.records],
        )
        self.assertEqual(response.trace_id, tracer.records[0]["trace_id"])
        self.assertEqual("chain", tracer.records[0]["as_type"])
        self.assertEqual("retriever", tracer.records[1]["as_type"])
        self.assertEqual("generation", tracer.records[2]["as_type"])
        self.assertEqual(
            response.answer,
            tracer.records[0]["updates"][-1]["output"]["answer"],
        )

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
        tracer = RecordingTracer()
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

        pipeline = RAGPipeline.from_config(
            config,
            registries=registries,
            tracer=tracer,
        )

        self.assertEqual("RAGLight is modular.", pipeline.generate("Question").answer)
        self.assertEqual("raglight.rag", tracer.records[0]["name"])


if __name__ == "__main__":
    unittest.main()
