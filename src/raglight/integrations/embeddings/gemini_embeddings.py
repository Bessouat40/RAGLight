from __future__ import annotations
import os
from typing import Optional, List

try:
    from typing_extensions import override
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_google_genai"),
        extra="gemini",
        feature="Google Gemini embeddings support",
    )

from ._base import EmbeddingAdapter


class GeminiEmbeddings(EmbeddingAdapter):
    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        super().__init__(model_name, api_base)

    @override
    def load(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
