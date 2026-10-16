"""Run classic RAG with OpenAI models and an ephemeral Chroma store."""

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.openai import OpenAIEmbeddings, OpenAILLM
from raglight.models import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="What does RAGLight v4 inject into a classic pipeline?",
    )
    parser.add_argument("--chat-model", default="gpt-5.6-terra")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = OpenAIEmbeddings(model_name=args.embedding_model)
    llm = OpenAILLM(model_name=args.chat_model)

    with TemporaryDirectory() as database:
        vector_store = ChromaVectorStore(
            collection_name="classic-openai-example",
            embeddings_model=embeddings,
            persist_directory=database,
        )
        vector_store.add_documents(
            [
                Document(
                    page_content=(
                        "A RAGLight classic pipeline receives an LLM, an embedding "
                        "model, and a vector store through dependency injection."
                    ),
                    metadata={"source": "inline-example"},
                )
            ]
        )
        pipeline = RAGPipeline(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
            config=PipelineConfig(top_k=3),
        )
        response = pipeline.generate(args.question)

    print(response.answer)
    print("Sources:", [source.metadata for source in response.sources])
    print("Trace:", response.trace_id)


if __name__ == "__main__":
    main()
