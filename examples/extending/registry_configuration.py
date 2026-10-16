"""Build a classic pipeline from named factories and serialized values."""

from __future__ import annotations

import argparse

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import IntegrationRegistries, IntegrationRegistry
from raglight.models import Document, RetrievalFilter


class ExampleLLM:
    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix

    def generate(self, input: dict[str, object]) -> str:
        return self.prefix + str(input["question"])

    def generate_streaming(self, input: dict[str, object]):
        yield self.generate(input)


class ExampleEmbeddings:
    def __init__(self, *, dimensions: int = 1) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] * self.dimensions for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] * self.dimensions


class ExampleVectorStore:
    def __init__(self, *, embeddings_model: object, content: str) -> None:
        self.embeddings_model = embeddings_model
        self.content = content

    def similarity_search(
        self,
        question: str,
        k: int = 5,
        *,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[Document]:
        if retrieval_filter is not None:
            raise NotImplementedError("this example store does not support filtering")
        return [Document(page_content=self.content)][:k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="How does named construction work?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registries = IntegrationRegistries(
        llms=IntegrationRegistry("LLM"),
        embeddings=IntegrationRegistry("embedding"),
        vector_stores=IntegrationRegistry("vector store"),
        rerankers=IntegrationRegistry("reranker"),
        document_processors=IntegrationRegistry("document processor"),
    )
    registries.llms.register("example", ExampleLLM)
    registries.embeddings.register("example", ExampleEmbeddings)
    registries.vector_stores.register("example", ExampleVectorStore)

    config = PipelineConfig.from_dict(
        {
            "top_k": 1,
            "llm": {"name": "example", "options": {"prefix": "Generated: "}},
            "embeddings": {
                "name": "example",
                "options": {"dimensions": 2},
            },
            "vector_store": {
                "name": "example",
                "options": {"content": "Registry-backed context."},
            },
        }
    )
    pipeline = RAGPipeline.from_config(config, registries=registries)
    response = pipeline.generate(args.question)
    print(response.answer)
    print(response.to_dict())


if __name__ == "__main__":
    main()
