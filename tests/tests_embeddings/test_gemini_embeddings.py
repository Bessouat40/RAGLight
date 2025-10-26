import unittest
from raglight.embeddings.gemini_embeddings import GeminiEmbeddingsModel
from ..test_config import TestsConfig


class TestGeminiEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embeddings = GeminiEmbeddingsModel(TestsConfig.GEMINI_EMBEDDING_MODEL)

    def test_model_load(self):
        self.assertTrue(
            self.embeddings.model is not None, "Model should be loaded successfully."
        )


if __name__ == "__main__":
    unittest.main()
