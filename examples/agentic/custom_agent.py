"""Inject a framework-free custom agent into the bounded agentic runtime."""

from __future__ import annotations

import argparse

from raglight.agentic import AgentContext, AgentDecision, AgenticPipeline
from raglight.models import Document


class CustomAgent:
    def decide(self, context: AgentContext) -> AgentDecision:
        if not context.steps:
            return AgentDecision.call_tool(
                "retrieve",
                {"query": context.question, "top_k": 2},
            )

        observation = context.steps[-1].observation
        if observation is None or observation.is_error:
            return AgentDecision.final("Retrieval did not return usable context.")
        return AgentDecision.final(
            f"The custom agent used this context: {observation.content}"
        )


class ExampleLLM:
    def generate(self, input: dict[str, object]) -> str:
        return "The custom agent owns decisions, so this LLM is not called."

    def generate_streaming(self, input: dict[str, object]):
        yield self.generate(input)


class ExampleEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class ExampleVectorStore:
    def similarity_search(self, question: str, k: int = 5) -> list[Document]:
        return [
            Document(
                page_content="A custom agent only implements decide(context).",
                metadata={"source": "custom-agent-example"},
            )
        ][:k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="What must a custom RAGLight agent implement?",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = AgenticPipeline(
        llm=ExampleLLM(),
        embeddings=ExampleEmbeddings(),
        vector_store=ExampleVectorStore(),
        agent=CustomAgent(),
        max_steps=3,
    )
    response = pipeline.generate(args.question)
    print(response.answer)
    print("Sources:", [source.metadata for source in response.sources])


if __name__ == "__main__":
    main()
