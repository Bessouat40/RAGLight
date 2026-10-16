from __future__ import annotations
from typing import Optional, List, Dict, Any

try:
    from typing_extensions import override
    from langchain_ollama import OllamaEmbeddings as LangChainOllamaEmbeddings
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_ollama"),
        extra="ollama",
        feature="Ollama embeddings support",
    )

from ._base import EmbeddingAdapter


class OllamaEmbeddings(EmbeddingAdapter):
    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_api_base = api_base or "http://localhost:11434"
        super().__init__(model_name, api_base=resolved_api_base)
        self.options = options or {}
        if "num_batch" not in self.options:
            self.options["num_batch"] = 8192
        if "num_ctx" not in self.options:
            self.options["num_ctx"] = 8192

    @override
    def load(self) -> LangChainOllamaEmbeddings:
        return LangChainOllamaEmbeddings(
            model=self.model_name,
            base_url=self.api_base,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
