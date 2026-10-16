"""Contract tests for custom v4 agents and agent tools."""

from __future__ import annotations

import unittest

from raglight.agentic import (
    Agent,
    AgentContext,
    AgentDecision,
    AgentTool,
    ToolResult,
)


class CustomAgent:
    def decide(self, context):
        return AgentDecision.final(context.question)


class CustomTool:
    name = "custom"
    description = "A custom structural tool."
    input_schema = {"type": "object"}

    def run(self, arguments):
        return ToolResult(str(arguments))


class AgenticContractTests(unittest.TestCase):
    def test_custom_agent_and_tool_conform_without_inheritance(self):
        agent = CustomAgent()
        tool = CustomTool()

        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(tool, AgentTool)
        self.assertNotIn(Agent, type(agent).__mro__)
        self.assertNotIn(AgentTool, type(tool).__mro__)
        self.assertEqual("question", agent.decide(AgentContext("question")).answer)
        self.assertEqual("{'value': 1}", tool.run({"value": 1}).content)


if __name__ == "__main__":
    unittest.main()
