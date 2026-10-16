"""Run classic RAG against a custom OpenAI-compatible HTTP endpoint."""

from __future__ import annotations

import argparse
import os
from tempfile import TemporaryDirectory

from raglight import RAGPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.openai import OpenAIEmbeddings, OpenAILLM
from raglight.models import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "not-required"),
        help="API key or a placeholder when the local server does not require one.",
    )
    parser.add_argument(
        "--question",
        default="What can an OpenAI-compatible endpoint provide to RAGLight?",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = OpenAIEmbeddings(
        model_name=args.embedding_model,
        api_base=args.base_url,
        api_key=args.api_key,
    )
    llm = OpenAILLM(
        model_name=args.chat_model,
        api_base=args.base_url,
        api_key=args.api_key,
    )

    with TemporaryDirectory() as database:
        vector_store = ChromaVectorStore(
            collection_name="openai-compatible-example",
            embeddings_model=embeddings,
            persist_directory=database,
        )
        vector_store.add_documents(
            [
                Document(
                    page_content=(
                        "RAGLight can use custom OpenAI-compatible chat and "
                        "embedding endpoints through the OpenAI adapters."
                    ),
                    metadata={"source": "inline-example"},
                )
            ]
        )
        pipeline = RAGPipeline(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
        )
        response = pipeline.generate(args.question)

    print(response.answer)
    print("Sources:", [source.metadata for source in response.sources])


if __name__ == "__main__":
    main()
