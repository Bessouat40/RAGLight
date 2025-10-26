import unittest
from raglight.embeddings.huggingface_embeddings import HuggingfaceEmbeddingsModel
from ..test_config import TestsConfig


class TestHuggingFaceEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embeddings = HuggingfaceEmbeddingsModel(TestsConfig.HUGGINGFACE_EMBEDDINGS)
          
    def test_model_load(self):
        self.assertTrue(
            self.embeddings.model is not None, "Model should be loaded successfully."
        )


if __name__ == "__main__":
    unittest.main()
