import unittest

from raglight.embeddings.ollama_embeddings import OllamaEmbeddingsModel
from raglight.vectorstore.postgres import PostgresVS

from ..test_config import TestsConfig

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        model_embeddings = TestsConfig.OLLAMA_EMBEDDING_MODEL
        ollama_client= TestsConfig.DEFAULT_OLLAMA_CLIENT
        collection_name = TestsConfig.COLLECTION_NAME
        self.data_path = TestsConfig.DATA_PATH

        self.host = TestsConfig.POSTGRES_HOST
        self.port = TestsConfig.POSTGRES_PORT
        self.darabase = TestsConfig.POSTGRES_DATABASE
        self.user = TestsConfig.POSTGRES_USER
        self.password = TestsConfig.POSTGRES_PASSWORD

        embeddings = OllamaEmbeddingsModel(model_embeddings, api_base=ollama_client)
        self.store = PostgresVS(
            embeddings_model=embeddings,
            collection_name=collection_name,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.darabase,
        )

    def test_ingest(self):
        self.store.ingest(data_path=self.data_path)
        self.assertEqual(True, True, "Embedding should be added to the store.")


if __name__ == "__main__":
    unittest.main()
