"""Unit tests for bounded v4 agentic orchestration."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest

from raglight.agentic import (
    AgentContext,
    AgentDecision,
    AgentStepLimitError,
    AgenticPipeline,
    FunctionTool,
    ToolResult,
)
from raglight.models import Document


class FakeLLM:
    def generate(self, input):
        return '{"type":"final","answer":"built-in answer"}'

    def generate_streaming(self, input):
        yield self.generate(input)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0]


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search(self, question, k=5):
        self.calls.append((question, k))
        return [
            Document(page_content="first", metadata={"rank": 1}),
            Document(page_content="second", metadata={"rank": 2}),
        ][:k]


class RecordingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, question, documents, top_k):
        self.calls.append((question, list(documents), top_k))
        return list(reversed(documents))[:top_k]


class ScriptedAgent:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.contexts: list[AgentContext] = []

    def decide(self, context):
        self.contexts.append(context)
        return self.decisions.pop(0)


class AgenticPipelineTests(unittest.TestCase):
    def build_pipeline(self, *, agent, **overrides):
        options = {
            "llm": FakeLLM(),
            "embeddings": FakeEmbeddings(),
            "vector_store": FakeVectorStore(),
            "agent": agent,
            "max_steps": 4,
        }
        options.update(overrides)
        return AgenticPipeline(**options)

    def test_custom_agent_can_call_a_custom_tool_then_finish(self):
        agent = ScriptedAgent(
            AgentDecision.call_tool("weather", {"city": "Paris"}),
            AgentDecision.final("It is sunny."),
        )
        calls = []
        tool = FunctionTool(
            name="weather",
            description="Return weather for a city.",
            handler=lambda arguments: calls.append(arguments) or "sunny",
            input_schema={"type": "object"},
        )
        pipeline = self.build_pipeline(agent=agent, tools=[tool])

        response = pipeline.generate(
            "What is the weather?",
            history=[{"role": "user", "content": "I live in Paris"}],
        )

        self.assertEqual("It is sunny.", response.answer)
        self.assertEqual([{"city": "Paris"}], calls)
        self.assertEqual(
            ["agent_decision", "tool", "agent_decision"],
            [event.stage for event in response.events],
        )
        self.assertEqual("weather", response.events[1].metadata["tool_name"])
        self.assertEqual("final", response.metadata["termination"])
        self.assertEqual(2, response.metadata["steps"])
        self.assertEqual("I live in Paris", agent.contexts[0].history[0]["content"])
        observation = agent.contexts[1].steps[0].observation
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual("sunny", observation.content)
        self.assertFalse(observation.is_error)

    def test_retrieval_is_a_builtin_tool_and_can_rerank_sources(self):
        agent = ScriptedAgent(
            AgentDecision.call_tool("retrieve", {"query": "architecture"}),
            AgentDecision.final("The second document is most relevant."),
        )
        store = FakeVectorStore()
        reranker = RecordingReranker()
        pipeline = self.build_pipeline(
            agent=agent,
            vector_store=store,
            reranker=reranker,
            top_k=2,
        )

        response = pipeline.generate("Explain the architecture")

        self.assertEqual([("architecture", 2)], store.calls)
        self.assertEqual("architecture", reranker.calls[0][0])
        self.assertEqual(
            ["second", "first"], [source.page_content for source in response.sources]
        )
        observation = agent.contexts[1].steps[0].observation
        assert observation is not None
        self.assertIn("second", observation.content)
        self.assertEqual(2, observation.metadata["retrieved_count"])
        self.assertTrue(observation.metadata["reranked"])

    def test_tool_failure_is_a_structured_observation_the_agent_can_recover_from(self):
        agent = ScriptedAgent(
            AgentDecision.call_tool("unstable", {}),
            AgentDecision.final("I recovered."),
        )

        def fail(_arguments):
            raise RuntimeError("service unavailable")

        pipeline = self.build_pipeline(
            agent=agent,
            tools=[FunctionTool("unstable", "Always fails.", fail)],
        )

        response = pipeline.generate("Try the tool")

        self.assertEqual("I recovered.", response.answer)
        self.assertEqual("failed", response.events[1].status)
        observation = agent.contexts[1].steps[0].observation
        assert observation is not None
        self.assertTrue(observation.is_error)
        self.assertIn("service unavailable", observation.content)

    def test_unknown_tool_is_reported_to_the_agent_without_execution(self):
        agent = ScriptedAgent(
            AgentDecision.call_tool("missing", {}),
            AgentDecision.final("Fallback answer."),
        )
        pipeline = self.build_pipeline(agent=agent)

        response = pipeline.generate("Use a missing tool")

        self.assertEqual("failed", response.events[1].status)
        observation = agent.contexts[1].steps[0].observation
        assert observation is not None
        self.assertTrue(observation.is_error)
        self.assertIn("Unknown agent tool 'missing'", observation.content)
        self.assertIn("retrieve", observation.content)

    def test_step_limit_raises_an_actionable_error_with_completed_events(self):
        agent = ScriptedAgent(
            AgentDecision.call_tool("retrieve", {"query": "one"}),
        )
        pipeline = self.build_pipeline(agent=agent, max_steps=1)

        with self.assertRaises(AgentStepLimitError) as raised:
            pipeline.generate("Never finishes")

        self.assertEqual(1, raised.exception.max_steps)
        self.assertEqual(2, len(raised.exception.events))
        self.assertTrue(raised.exception.trace_id)
        self.assertIn("maximum of 1 step", str(raised.exception))

    def test_constructor_and_questions_are_validated_early(self):
        agent = ScriptedAgent(AgentDecision.final("answer"))

        with self.assertRaisesRegex(ValueError, "max_steps must be greater than zero"):
            self.build_pipeline(agent=agent, max_steps=0)
        with self.assertRaisesRegex(ValueError, "top_k must be greater than zero"):
            self.build_pipeline(agent=agent, top_k=0)

        pipeline = self.build_pipeline(agent=agent)
        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            pipeline.generate("   ")

    def test_duplicate_tool_names_are_rejected_case_insensitively(self):
        first = FunctionTool("Search", "First.", lambda _: "one")
        second = FunctionTool(" search ", "Second.", lambda _: "two")

        with self.assertRaisesRegex(
            ValueError, "agent tool 'search' is already registered"
        ):
            self.build_pipeline(
                agent=ScriptedAgent(AgentDecision.final("answer")),
                tools=[first, second],
            )

    def test_retrieval_name_is_reserved_for_the_builtin_tool(self):
        custom_retrieval = FunctionTool("retrieve", "Override.", lambda _: "no")

        with self.assertRaisesRegex(
            ValueError, "agent tool 'retrieve' is already registered"
        ):
            self.build_pipeline(
                agent=ScriptedAgent(AgentDecision.final("answer")),
                tools=[custom_retrieval],
            )

    def test_async_agents_and_tools_are_supported_by_agenerate(self):
        class AsyncAgent:
            async def decide(self, context):
                if not context.steps:
                    return AgentDecision.call_tool("async", {"value": "ready"})
                observation = context.steps[-1].observation
                assert observation is not None
                return AgentDecision.final(observation.content)

        async def handler(arguments):
            return ToolResult(f"async-{arguments['value']}")

        pipeline = self.build_pipeline(
            agent=AsyncAgent(),
            tools=[FunctionTool("async", "Async tool.", handler)],
        )

        response = asyncio.run(pipeline.agenerate("Run asynchronously"))

        self.assertEqual("async-ready", response.answer)

    def test_sync_generate_has_actionable_error_inside_an_event_loop(self):
        pipeline = self.build_pipeline(
            agent=ScriptedAgent(AgentDecision.final("answer"))
        )

        async def call_sync_api():
            with self.assertRaisesRegex(RuntimeError, "await pipeline.agenerate"):
                pipeline.generate("question")

        asyncio.run(call_sync_api())


class AgenticSessionTests(unittest.TestCase):
    def test_sessions_keep_conversation_history_isolated(self):
        class HistoryAgent:
            def __init__(self):
                self.histories = []

            def decide(self, context):
                self.histories.append(
                    tuple(dict(message) for message in context.history)
                )
                return AgentDecision.final(f"answer-{len(self.histories)}")

        agent = HistoryAgent()
        pipeline = AgenticPipeline(
            llm=FakeLLM(),
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            agent=agent,
        )
        first = pipeline.create_session(session_id="first")
        second = pipeline.create_session(session_id="second")

        first.generate("first question")
        second.generate("second question")
        first.generate("follow up")

        self.assertEqual((), agent.histories[0])
        self.assertEqual((), agent.histories[1])
        self.assertEqual(
            (
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer-1"},
            ),
            agent.histories[2],
        )
        self.assertEqual(4, len(first.history))
        self.assertEqual(2, len(second.history))

    def test_concurrent_sessions_do_not_share_history_or_a_global_lock(self):
        barrier = Barrier(2)

        class ConcurrentAgent:
            def decide(self, context):
                barrier.wait(timeout=2)
                return AgentDecision.final(f"{context.question}:{len(context.history)}")

        pipeline = AgenticPipeline(
            llm=FakeLLM(),
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            agent=ConcurrentAgent(),
        )
        first = pipeline.create_session(session_id="first")
        second = pipeline.create_session(session_id="second")

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(first.generate, "alpha")
            second_future = executor.submit(second.generate, "beta")

        self.assertEqual("alpha:0", first_future.result().answer)
        self.assertEqual("beta:0", second_future.result().answer)
        self.assertEqual("alpha", first.history[0]["content"])
        self.assertEqual("beta", second.history[0]["content"])

    def test_failed_turn_is_not_committed_to_a_session(self):
        agent = ScriptedAgent(
            AgentDecision.call_tool("retrieve", {"query": "never final"})
        )
        pipeline = AgenticPipeline(
            llm=FakeLLM(),
            embeddings=FakeEmbeddings(),
            vector_store=FakeVectorStore(),
            agent=agent,
            max_steps=1,
        )
        session = pipeline.create_session(session_id="failure")

        with self.assertRaises(AgentStepLimitError):
            session.generate("failed question")

        self.assertEqual((), session.history)


if __name__ == "__main__":
    unittest.main()
