"""Expose one remote MCP tool to Agentic RAG over streamable HTTP."""

from __future__ import annotations

import argparse
import asyncio
from tempfile import TemporaryDirectory

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from raglight.agentic import AgenticPipeline
from raglight.agentic.mcp import MCPTool
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.models import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--tool", required=True, help="MCP tool name to expose.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--chat-model", default="qwen3:0.6b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    embeddings = OllamaEmbeddings(
        model_name=args.embedding_model,
        api_base=args.ollama_url,
    )
    llm = OllamaLLM(
        model_name=args.chat_model,
        api_base=args.ollama_url,
        preload_model=False,
    )

    async with streamable_http_client(args.server_url) as streams:
        read_stream, write_stream = streams
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            listed_tools = await client.list_tools()
            selected = next(
                (tool for tool in listed_tools.tools if tool.name == args.tool),
                None,
            )
            if selected is None:
                available = ", ".join(tool.name for tool in listed_tools.tools)
                raise ValueError(
                    f"MCP tool '{args.tool}' was not found. Available: {available}"
                )

            tool = MCPTool(
                client=client,
                name=selected.name,
                description=selected.description or f"Call {selected.name}.",
                input_schema=selected.input_schema,
            )
            with TemporaryDirectory() as database:
                vector_store = ChromaVectorStore(
                    collection_name="agentic-mcp-example",
                    embeddings_model=embeddings,
                    persist_directory=database,
                )
                vector_store.add_documents(
                    [
                        Document(
                            page_content=(
                                "The connected MCP server provides external tools; "
                                "RAGLight owns the bounded agent loop."
                            ),
                            metadata={"source": "inline-example"},
                        )
                    ]
                )
                pipeline = AgenticPipeline(
                    llm=llm,
                    embeddings=embeddings,
                    vector_store=vector_store,
                    tools=[tool],
                    max_steps=5,
                )
                response = await pipeline.agenerate(args.question)

    print(response.answer)
    print("Events:", [event.to_dict() for event in response.events])


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
