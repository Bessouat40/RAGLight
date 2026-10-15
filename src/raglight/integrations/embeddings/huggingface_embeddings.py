from __future__ import annotations
from typing import List

try:
    from typing_extensions import override
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "sentence_transformers"),
        extra="huggingface",
        feature="Hugging Face embeddings support",
    )

from ._base import EmbeddingAdapter


class HuggingFaceEmbeddings(EmbeddingAdapter):
    """
    Concrete implementation of the EmbeddingsModel for HuggingFace models using sentence-transformers.
    This runs locally.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a HuggingFaceEmbeddings instance.

        Args:
            model_name (str): The name of the HuggingFace model to load locally.
        """
        super().__init__(model_name)

    @override
    def load(self) -> SentenceTransformer:
        """
        Loads the SentenceTransformer model locally.

        Returns:
            SentenceTransformer: The loaded model.
        """
        return SentenceTransformer(self.model_name)

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
