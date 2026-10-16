"""Run bounded Agentic RAG with local Ollama models and Chroma."""

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory

from raglight.agentic import AgenticPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.models import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="How are classic and Agentic RAG separated in RAGLight v4?",
    )
    parser.add_argument("--chat-model", default="qwen3:0.6b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-steps", type=int, default=4)
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
            collection_name="agentic-ollama-example",
            embeddings_model=embeddings,
            persist_directory=database,
        )
        vector_store.add_documents(
            [
                Document(
                    page_content=(
                        "RAGLight v4 exposes classic RAG through RAGPipeline and "
                        "bounded Agentic RAG through a separate AgenticPipeline."
                    ),
                    metadata={"source": "inline-example"},
                )
            ]
        )
        pipeline = AgenticPipeline(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
            max_steps=args.max_steps,
        )
        response = pipeline.generate(args.question)

    print(response.answer)
    print("Sources:", [source.metadata for source in response.sources])
    print("Events:", [event.to_dict() for event in response.events])


if __name__ == "__main__":
    main()
