"""Run RAGLight with local Ollama inference and anonymous Parallel MCP tools."""

import argparse

from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.settings import Settings
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline


def create_config(
    model: str = "qwen3:8b", api_base: str = "http://localhost:11434"
) -> AgenticRAGConfig:
    """Create an opt-in connection without reading or sending Parallel credentials."""
    return AgenticRAGConfig(
        provider=Settings.OLLAMA,
        model=model,
        api_base=api_base,
        knowledge_base=[],
        max_steps=12,
        system_prompt=(
            "Use web_search for current web information and web_fetch to read relevant "
            "pages. Cite source URLs in your answer. The local knowledge base is empty."
        ),
        mcp_config={
            "parallel": {
                "url": "https://search.parallel.ai/mcp",
                "transport": "streamable_http",
                "headers": {
                    "User-Agent": "RAGLight-Parallel-MCP/1.0 (https://github.com/Bessouat40/RAGLight)"
                },
            }
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="Find the current Python release, read its release page and cite the URL.",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--persist-directory", default="./parallel-example-db")
    args = parser.parse_args()

    config = create_config(args.model, args.api_base)
    vector_store_config = VectorStoreConfig(
        provider=Settings.OLLAMA,
        embedding_model=args.embedding_model,
        api_base=args.api_base,
        database=Settings.QDRANT,
        persist_directory=args.persist_directory,
        collection_name="parallel_example",
    )
    with AgenticRAGPipeline(config, vector_store_config) as pipeline:
        pipeline.build()
        print(pipeline.generate(args.question))


if __name__ == "__main__":
    main()
