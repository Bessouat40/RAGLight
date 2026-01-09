from __future__ import annotations
from typing import Optional, List, Any
from typing_extensions import override

from google.genai import Client
from google.genai import types

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel


class GeminiEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for Gemini models using the official Google GenAI library.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        """
        Initializes a GeminiEmbeddingsModel instance.

        Args:
            model_name (str): The name of the Gemini model to load (e.g., "models/embedding-001").
            api_base (Optional[str]): Not strictly used by the official lib as it relies on global config,
                                      but kept for interface consistency.
        """
        super().__init__(model_name, api_base)

    @override
    def load(self) -> Any:
        """
        Configures the Google GenAI library.
        Returns the module reference as the 'client'.
        """
        return Client(
            api_key=Settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(base_url=self.api_base),
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents using Google GenAI.
        Specifies 'retrieval_document' task type for optimized document storage embeddings.
        """
        result = self.model.embed_content(
            model=self.model_name, content=texts, task_type="retrieval_document"
        )
        return result["embedding"]

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        Specifies 'retrieval_query' task type for optimized search query embeddings.
        """
        result = self.model.embed_content(
            model=self.model_name, content=text, task_type="retrieval_query"
        )
        return result["embedding"]
