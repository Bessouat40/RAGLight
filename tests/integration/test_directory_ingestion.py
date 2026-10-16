"""Integration coverage for built-in document processing and public ingestion."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from raglight import RAGPipeline
from raglight.integrations import default_registries
from raglight.models import Document, FolderSource


class FakeLLM:
    def generate(self, input):
        return "answer"

    def generate_streaming(self, input):
        yield self.generate(input)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class InMemoryStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def similarity_search(self, question, k=5):
        matches = [
            document
            for document in self.documents
            if question.lower() in document.page_content.lower()
        ]
        return matches[:k]


class DirectoryIngestionIntegrationTests(unittest.TestCase):
    def test_mixed_nested_directory_uses_builtin_processors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "overview.txt").write_text(
                "RAGLight supports public directory ingestion.", encoding="utf-8"
            )
            nested = root / "guides"
            nested.mkdir()
            (nested / "usage.md").write_text(
                "Nested Markdown is processed automatically.", encoding="utf-8"
            )
            (nested / "archive.bin").write_bytes(b"unsupported")
            store = InMemoryStore()
            pipeline = RAGPipeline(
                llm=FakeLLM(),
                embeddings=FakeEmbeddings(),
                vector_store=store,
                document_processors=default_registries.document_processors,
            )

            result = pipeline.ingest([FolderSource(path=root)])

        self.assertEqual(3, result.discovered_count)
        self.assertEqual(2, result.processed_count)
        self.assertEqual(2, result.indexed_count)
        self.assertEqual(1, result.skipped_count)
        self.assertEqual(0, result.failed_count)
        self.assertEqual(2, len(store.documents))
        self.assertEqual(
            {str(root / "overview.txt"), str(nested / "usage.md")},
            {document.metadata["source"] for document in store.documents},
        )


if __name__ == "__main__":
    unittest.main()
