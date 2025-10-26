from __future__ import annotations
from typing import Any, Optional
from typing_extensions import override

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel
from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for Ollama models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using Ollama embeddings.

    Attributes:
        model_name (str): The name of the Ollama model to be loaded.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        """
        Initializes a OllamaEmbeddingsModel instance.

        Args:
            model_name (str): The name of the Ollama model to load.
        """
        self.api_base = api_base or Settings.DEFAULT_OLLAMA_CLIENT
        super().__init__(model_name)

    @override
    def load(self) -> OllamaEmbeddings:
        """
        Loads the Ollama embeddings model.

        This method overrides the abstract `load` method from the `EmbeddingsModel` class
        and initializes the Ollama embeddings model with the specified `model_name`.

        Returns:
            OllamaEmbeddings: The loaded Ollama embeddings model.
        """
        return OllamaEmbeddings(model=self.model_name, base_url=self.api_base)
    
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents using the loaded Ollama embeddings model.

        Args:
            documents (list[str]): List of documents (strings) to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        return self.model.embed_documents(documents)
    
    def embed_query(self, question) -> str:
        """
        Embeds a list of documents using the loaded Ollama embeddings model.

        Args:
            documents (list[str]): List of documents (strings) to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        return self.model.embed_query(question)