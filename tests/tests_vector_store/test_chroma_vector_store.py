import shutil
import tempfile
import unittest

from raglight import RAGPipeline
from raglight.integrations import default_registries
from raglight.integrations.chroma import ChromaVectorStore
from raglight.models import FolderSource

from ..test_config import TestsConfig


class DeterministicEmbeddings:
    """Small offline test double with stable two-dimensional embeddings."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(sum(map(ord, text)) % 997)]


class StubLLM:
    def generate(self, input):
        return "answer"

    def generate_streaming(self, input):
        yield self.generate(input)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.persist_directory = tempfile.mkdtemp(prefix="raglight-chroma-test-")
        self.data_path = tempfile.mkdtemp(prefix="raglight-ingestion-test-")
        with open(f"{self.data_path}/document.txt", "w", encoding="utf-8") as document:
            document.write("RAGLight provides deterministic retrieval workflows.")
        collection_name = TestsConfig.COLLECTION_NAME
        embeddings = DeterministicEmbeddings()
        self.store = ChromaVectorStore(
            embeddings_model=embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name,
        )

    def tearDown(self):
        shutil.rmtree(self.persist_directory)
        shutil.rmtree(self.data_path)

    def test_ingest(self):
        pipeline = RAGPipeline(
            llm=StubLLM(),
            embeddings=self.store.embeddings_model,
            vector_store=self.store,
            document_processors=default_registries.document_processors,
        )

        result = pipeline.ingest([FolderSource(path=self.data_path)])

        self.assertEqual(1, result.processed_count)
        self.assertGreater(self.store.collection.count(), 0)
        self.assertFalse(hasattr(self.store, "ingest"))


if __name__ == "__main__":
    unittest.main()
