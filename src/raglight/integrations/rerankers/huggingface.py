"""Hugging Face reranker integration."""

from __future__ import annotations

from ...models import Document

try:
    from sentence_transformers import CrossEncoder
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency="sentence_transformers",
        extra="huggingface",
        feature="Hugging Face reranking support",
    )


class HuggingFaceReranker:
    """Rerank documents with a sentence-transformers cross encoder."""

    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        if not documents:
            return []
        results = self.model.rank(
            query=question,
            documents=[document.page_content for document in documents],
            top_k=top_k,
            return_documents=False,
        )
        return [documents[result["corpus_id"]] for result in results]
