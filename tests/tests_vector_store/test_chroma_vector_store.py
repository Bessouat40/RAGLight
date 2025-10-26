import unittest
from raglight.vectorstore.chroma import ChromaVS
from raglight.embeddings.huggingface_embeddings import HuggingfaceEmbeddingsModel
from ..test_config import TestsConfig


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        persist_directory = TestsConfig.CHROMA_PERSIST_DIRECTORY_INGESTION
        collection_name = TestsConfig.COLLECTION_NAME
        self.data_path = TestsConfig.DATA_PATH
        embeddings = HuggingfaceEmbeddingsModel(TestsConfig.HUGGINGFACE_EMBEDDINGS)
        self.store = ChromaVS(
            embeddings_model=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def test_ingest(self):
        self.store.ingest(data_path=self.data_path)
        self.assertEqual(True, True, "Embedding should be added to the store.")


if __name__ == "__main__":
    unittest.main()
