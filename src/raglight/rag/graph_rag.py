from __future__ import annotations

import json
from typing import Any, Dict
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langchain_community.graphs import Neo4jGraph

from ..llm.llm import LLM


class GraphState(TypedDict):
    """Represents the state of the Graph RAG process."""

    question: str
    cypher_query: str
    graph_result: str
    answer: str


class GraphRAG:
    """Graph-based Retrieval-Augmented Generation powered by Neo4j."""

    DEFAULT_CYPHER_PROMPT: str = (
        "You are an expert Neo4j engineer. "
        "Generate a Cypher query that answers the user's question using the provided schema. "
        "Return only the Cypher query without explanations."
    )

    def __init__(
        self,
        llm: LLM,
        graph: Neo4jGraph,
        cypher_prompt: str | None = None,
        stream: bool = False,
    ) -> None:
        self.llm = llm
        self.graph = graph
        self.cypher_prompt = cypher_prompt or self.DEFAULT_CYPHER_PROMPT
        self.stream = stream
        self.workflow: Any = self._create_graph()

    def generate_cypher(self, state: Dict[str, str]) -> Dict[str, str]:
        prompt = {
            "instruction": self.cypher_prompt,
            "schema": self.graph.schema,
            "question": state["question"],
        }
        cypher_query = self.llm.generate(prompt)
        return {"cypher_query": cypher_query, "question": state["question"]}

    def run_cypher(self, state: Dict[str, str]) -> Dict[str, str]:
        result = self.graph.query(state["cypher_query"])
        return {
            "question": state["question"],
            "cypher_query": state["cypher_query"],
            "graph_result": self._format_graph_result(result),
        }

    def generate_answer(self, state: Dict[str, str]) -> Dict[str, Any]:
        prompt = {
            "question": state["question"],
            "cypher_query": state["cypher_query"],
            "graph_result": state["graph_result"],
        }
        if self.stream and hasattr(self.llm, "generate_streaming"):
            response = self.llm.generate_streaming(prompt)  # type: ignore[attr-defined]
        else:
            response = self.llm.generate(prompt)
        return {"answer": response}

    def _create_graph(self) -> Any:
        graph_builder = StateGraph(GraphState).add_sequence(
            [self.generate_cypher, self.run_cypher, self.generate_answer]
        )
        graph_builder.add_edge(START, "generate_cypher")
        return graph_builder.compile()

    def generate(self, question: str) -> Any:
        state = {"question": question}
        response = self.workflow.invoke(state)
        return response["answer"]

    @staticmethod
    def _format_graph_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            return "\n".join(json.dumps(row, default=str) for row in result)
        if isinstance(result, dict):
            return json.dumps(result, default=str)
        return str(result)
