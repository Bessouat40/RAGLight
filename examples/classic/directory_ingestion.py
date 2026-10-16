"""Recursively ingest a directory, then query it with Ollama and Chroma."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from raglight import RAGPipeline
from raglight.integrations import default_registries
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.models import FolderSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory to ingest recursively.")
    parser.add_argument(
        "--question",
        default="What are the main topics in these documents?",
    )
    parser.add_argument("--chat-model", default="qwen3:0.6b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = OllamaEmbeddings(
        model_name=args.embedding_model,
        api_base=args.ollama_url,
    )
    llm = OllamaLLM(
        model_name=args.chat_model,
        api_base=args.ollama_url,
        preload_model=False,
    )

    with TemporaryDirectory() as database:
        vector_store = ChromaVectorStore(
            collection_name="directory-ingestion-example",
            embeddings_model=embeddings,
            persist_directory=database,
        )
        pipeline = RAGPipeline(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
            document_processors=default_registries.document_processors,
        )
        ingestion = pipeline.ingest([FolderSource(path=args.directory)])
        response = pipeline.generate(args.question)

    print("Ingestion:", ingestion.to_dict())
    print(response.answer)
    print("Sources:", [source.metadata for source in response.sources])


if __name__ == "__main__":
    main()
