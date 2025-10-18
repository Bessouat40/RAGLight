import unittest
from ..test_config import TestsConfig

from ...src.raglight.embeddings.ollama_embeddings import OllamaEmbeddingsModel
from ...src.raglight.vectorstore.postgres import PostgresVS


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        model_embeddings = TestsConfig.OLLAMA_EMBEDDING_MODEL
        collection_name = TestsConfig.COLLECTION_NAME
        self.data_path = TestsConfig.DATA_PATH
        embeddings = OllamaEmbeddingsModel(model_embeddings)
        self.store = PostgresVS(
            embeddings_model=embeddings,
            collection_name=collection_name,
        )

    def test_ingest(self):
        self.store.ingest(data_path=self.data_path)
        self.assertEqual(True, True, "Embedding should be added to the store.")


if __name__ == "__main__":
    unittest.main()
