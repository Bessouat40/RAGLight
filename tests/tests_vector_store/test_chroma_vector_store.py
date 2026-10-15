import shutil
import tempfile
import unittest

from raglight.integrations.chroma import ChromaVectorStore

from ..test_config import TestsConfig


class DeterministicEmbeddings:
    """Small offline test double with stable two-dimensional embeddings."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(sum(map(ord, text)) % 997)]


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
        self.store.ingest(data_path=self.data_path)

        self.assertGreater(self.store.collection.count(), 0)


if __name__ == "__main__":
    unittest.main()
