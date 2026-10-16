"""Unit tests for agentic decisions, tools, and planner values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from raglight.agentic import (
    AgentContext,
    AgentDecision,
    AgentDecisionError,
    FunctionTool,
    LLMPlannerAgent,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class SequencedLLM:
    def __init__(self, response):
        self.response = response
        self.inputs = []

    def generate(self, input):
        self.inputs.append(input)
        return self.response

    def generate_streaming(self, input):
        yield self.generate(input)


class AgenticModelTests(unittest.TestCase):
    def test_decisions_enforce_exactly_one_supported_action(self):
        self.assertEqual("final", AgentDecision.final("answer").type)
        self.assertEqual("tool", AgentDecision.call_tool("search", {"q": "x"}).type)

        with self.assertRaisesRegex(ValueError, "final answer must not be empty"):
            AgentDecision.final(" ")
        with self.assertRaisesRegex(ValueError, "tool name must not be empty"):
            AgentDecision.call_tool(" ", {})
        with self.assertRaisesRegex(ValueError, "unsupported agent decision"):
            AgentDecision(type="other")

    def test_public_values_are_immutable_snapshots(self):
        schema = {"type": "object"}
        spec = ToolSpec("search", "Search docs.", schema)
        schema["type"] = "changed"
        result = ToolResult("found", metadata={"count": 1})
        context = AgentContext(question="question", tools=[spec])

        self.assertEqual("object", spec.input_schema["type"])
        with self.assertRaises(TypeError):
            result.metadata["count"] = 2
        with self.assertRaises(FrozenInstanceError):
            context.question = "changed"

    def test_function_tool_normalizes_string_and_structured_results(self):
        string_tool = FunctionTool("echo", "Echo input.", lambda args: args["value"])
        structured = ToolResult("done", metadata={"ok": True})
        result_tool = FunctionTool("result", "Return result.", lambda _: structured)

        self.assertEqual("hello", string_tool.run({"value": "hello"}).content)
        self.assertIs(structured, result_tool.run({}))

    def test_tool_registry_has_deterministic_specs_and_actionable_lookup(self):
        registry = ToolRegistry()
        registry.register(FunctionTool("Zulu", "Last.", lambda _: "z"))
        registry.register(FunctionTool("alpha", "First.", lambda _: "a"))

        self.assertEqual(("alpha", "zulu"), registry.names)
        self.assertEqual(["alpha", "zulu"], [spec.name for spec in registry.specs])
        with self.assertRaisesRegex(
            LookupError,
            "Unknown agent tool 'missing'.*alpha, zulu",
        ):
            registry.get("missing")

    def test_default_llm_planner_parses_json_and_receives_only_public_context(self):
        llm = SequencedLLM(
            '```json\n{"type":"tool","tool_name":"search",'
            '"arguments":{"query":"RAG"}}\n```'
        )
        planner = LLMPlannerAgent(llm)
        context = AgentContext(
            question="What is RAG?",
            history=[{"role": "user", "content": "Earlier"}],
            tools=[ToolSpec("search", "Search docs.")],
        )

        decision = planner.decide(context)

        self.assertEqual(AgentDecision.call_tool("search", {"query": "RAG"}), decision)
        self.assertEqual("agentic_decision", llm.inputs[0]["mode"])
        self.assertEqual("What is RAG?", llm.inputs[0]["original_question"])
        self.assertIn("What is RAG?", llm.inputs[0]["question"])
        self.assertIn('"name": "search"', llm.inputs[0]["question"])
        self.assertEqual("search", llm.inputs[0]["tools"][0]["name"])

    def test_default_planner_accepts_a_json_object_after_model_reasoning(self):
        planner = LLMPlannerAgent(
            SequencedLLM(
                "I should use the knowledge base first.\n"
                '{"type":"tool","tool_name":"retrieve",'
                '"arguments":{"query":"RAGLight"}}'
            )
        )

        decision = planner.decide(AgentContext(question="question"))

        self.assertEqual("retrieve", decision.tool_name)

    def test_default_llm_planner_rejects_invalid_or_unsupported_output(self):
        for response, message in (
            ("not json", "valid JSON"),
            ('{"type":"unknown"}', "unsupported agent decision"),
            ('{"type":"tool","tool_name":"search","arguments":[]}', "arguments"),
        ):
            with self.subTest(response=response):
                planner = LLMPlannerAgent(SequencedLLM(response))
                with self.assertRaisesRegex(AgentDecisionError, message):
                    planner.decide(AgentContext(question="question"))


if __name__ == "__main__":
    unittest.main()
