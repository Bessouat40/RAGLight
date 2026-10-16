"""Run classic RAG entirely with framework-free custom components."""

from __future__ import annotations

import argparse

from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document


class CustomEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class CustomVectorStore:
    def similarity_search(self, question: str, k: int = 5) -> list[Document]:
        return [
            Document(
                page_content="RAGLight accepts structural custom components.",
                metadata={"source": "custom-store"},
            )
        ][:k]


class CustomLLM:
    def generate(self, input: dict[str, object]) -> str:
        return f"Custom answer for: {input['question']}"

    def generate_streaming(self, input: dict[str, object]):
        yield self.generate(input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="Can I inject custom components?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = RAGPipeline(
        llm=CustomLLM(),
        embeddings=CustomEmbeddings(),
        vector_store=CustomVectorStore(),
        config=PipelineConfig(top_k=1),
    )
    response = pipeline.generate(args.question)
    print(response.answer)
    print("Sources:", [source.metadata for source in response.sources])


if __name__ == "__main__":
    main()
