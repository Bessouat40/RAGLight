from __future__ import annotations
import os
from typing import Optional, List

try:
    from typing_extensions import override
    from langchain_openai import OpenAIEmbeddings
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_openai"),
        extra="openai",
        feature="OpenAI embeddings support",
    )

from ._base import EmbeddingAdapter


class OpenAIEmbeddings(EmbeddingAdapter):
    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_api_base = api_base or "https://api.openai.com/v1"
        super().__init__(model_name, api_base=resolved_api_base)

    @override
    def load(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.api_base,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
