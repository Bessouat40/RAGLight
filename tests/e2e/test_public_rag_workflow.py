import unittest
from dataclasses import dataclass

from raglight import PipelineConfig, RAGPipeline


@dataclass(frozen=True)
class Document:
    page_content: str


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

        answer = pipeline.generate("What is RAGLight?")

        self.assertEqual("RAGLight is modular.", answer)
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


if __name__ == "__main__":
    unittest.main()
