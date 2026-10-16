"""Bounded orchestration for the separate public agentic API."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
import inspect
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from ..contracts import EmbeddingModel, LLM, Reranker, VectorStore
from ..models import Document, ExecutionEvent, ExecutionStatus, RAGResponse
from .contracts import Agent, AgentTool
from .errors import AgentStepLimitError
from .models import (
    AgentContext,
    AgentDecision,
    AgentStep,
    ToolObservation,
    ToolResult,
)
from .planner import LLMPlannerAgent
from .session import AgenticSession
from .tools import RetrievalTool, ToolRegistry

ResultT = TypeVar("ResultT")


class AgenticPipeline:
    """Execute agent-selected tools within an explicit step budget."""

    def __init__(
        self,
        *,
        llm: LLM,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
        tools: Sequence[AgentTool] = (),
        agent: Agent | None = None,
        reranker: Reranker | None = None,
        max_steps: int = 4,
        top_k: int = 5,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be greater than zero")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        self._agent = agent or LLMPlannerAgent(llm)
        self._max_steps = max_steps
        self._top_k = top_k
        self._tools = ToolRegistry()
        self._tools.register(
            RetrievalTool(vector_store, top_k=top_k, reranker=reranker)
        )
        for tool in tools:
            self._tools.register(tool)

    def create_session(
        self,
        *,
        session_id: str | None = None,
        history: Sequence[Mapping[str, str]] = (),
    ) -> AgenticSession:
        return AgenticSession(self, session_id=session_id, history=history)

    def generate(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> RAGResponse:
        """Run the bounded loop from synchronous application code."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate(question, history=history))
        raise RuntimeError(
            "AgenticPipeline.generate() cannot run inside an active event loop; "
            "use 'await pipeline.agenerate(...)' instead."
        )

    async def agenerate(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> RAGResponse:
        """Run the bounded loop and await asynchronous agents or tools."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        trace_id = str(uuid4())
        total_started = perf_counter()
        steps: list[AgentStep] = []
        events: list[ExecutionEvent] = []
        timings: dict[str, float] = {}
        sources: list[Document] = []

        for step_number in range(1, self._max_steps + 1):
            context = AgentContext(
                question=normalized_question,
                history=history,
                tools=self._tools.specs,
                steps=steps,
            )
            decision_started = perf_counter()
            decision = await self._resolve(self._agent.decide(context))
            if not isinstance(decision, AgentDecision):
                raise TypeError("agents must return AgentDecision")
            decision_duration = self._elapsed_ms(decision_started)
            timings[f"decision_{step_number}"] = decision_duration
            events.append(
                ExecutionEvent(
                    stage="agent_decision",
                    status="completed",
                    duration_ms=decision_duration,
                    metrics={"step": step_number},
                    metadata={
                        "step": step_number,
                        "decision": decision.type,
                        **(
                            {"tool_name": decision.tool_name}
                            if decision.tool_name is not None
                            else {}
                        ),
                    },
                )
            )
            if decision.type == "final":
                steps.append(AgentStep(step_number, decision))
                timings["total"] = self._elapsed_ms(total_started)
                assert decision.answer is not None
                return RAGResponse(
                    answer=decision.answer,
                    sources=sources,
                    events=events,
                    timings=timings,
                    metadata={
                        "steps": step_number,
                        "max_steps": self._max_steps,
                        "termination": "final",
                        "tools_used": [
                            step.decision.tool_name
                            for step in steps
                            if step.decision.tool_name is not None
                        ],
                    },
                    trace_id=trace_id,
                )

            assert decision.tool_name is not None
            tool_started = perf_counter()
            try:
                tool = self._tools.get(decision.tool_name)
                result = await self._resolve(tool.run(decision.arguments))
                if not isinstance(result, ToolResult):
                    raise TypeError("agent tools must return ToolResult")
                observation = ToolObservation(
                    tool_name=decision.tool_name,
                    content=result.content,
                    sources=result.sources,
                    metadata=result.metadata,
                )
                sources.extend(result.sources)
                status: ExecutionStatus = "completed"
                message = f"Tool '{decision.tool_name}' completed."
                metrics = {"source_count": len(result.sources)}
            except Exception as error:
                observation = ToolObservation(
                    tool_name=decision.tool_name,
                    content=f"{type(error).__name__}: {error}",
                    metadata={"error_type": type(error).__name__},
                    is_error=True,
                )
                status = "failed"
                message = f"Tool '{decision.tool_name}' failed."
                metrics = {"source_count": 0}
            tool_duration = self._elapsed_ms(tool_started)
            timings[f"tool_{step_number}"] = tool_duration
            events.append(
                ExecutionEvent(
                    stage="tool",
                    status=status,
                    duration_ms=tool_duration,
                    metrics=metrics,
                    message=message,
                    metadata={
                        "step": step_number,
                        "tool_name": decision.tool_name,
                    },
                )
            )
            steps.append(AgentStep(step_number, decision, observation))

        raise AgentStepLimitError(
            max_steps=self._max_steps,
            trace_id=trace_id,
            events=events,
        )

    @staticmethod
    async def _resolve(value: ResultT | Awaitable[ResultT]) -> ResultT:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return max(0.0, (perf_counter() - started) * 1000)
