"""Integration tests for the optional MCP agent-tool adapter."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import unittest

from mcp.types import CallToolResult, TextContent

from raglight.agentic.mcp import MCPTool, MCPToolError


class FakeMCPClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return self.result


class MCPToolIntegrationTests(unittest.TestCase):
    def test_mcp_result_is_translated_to_the_agent_tool_contract(self):
        client = FakeMCPClient(
            CallToolResult(
                content=[TextContent(type="text", text="remote result")],
                structuredContent={"count": 1},
                isError=False,
            )
        )
        tool = MCPTool(
            client=client,
            name="remote_search",
            description="Search a remote MCP server.",
            input_schema={"type": "object"},
        )

        result = asyncio.run(tool.run({"query": "RAG"}))

        self.assertEqual("remote result", result.content)
        self.assertEqual({"count": 1}, result.metadata["structured_content"])
        self.assertEqual([("remote_search", {"query": "RAG"})], client.calls)

    def test_mcp_errors_are_actionable_and_do_not_expose_raw_objects(self):
        client = FakeMCPClient(
            CallToolResult(
                content=[TextContent(type="text", text="permission denied")],
                isError=True,
            )
        )
        tool = MCPTool(client, "remote", "Remote tool.")

        with self.assertRaisesRegex(MCPToolError, "remote.*permission denied"):
            asyncio.run(tool.run({}))

    def test_missing_mcp_dependency_names_the_installation_extra(self):
        script = textwrap.dedent("""
            import builtins

            original_import = builtins.__import__
            def blocked_import(name, *args, **kwargs):
                if name == "mcp" or name.startswith("mcp."):
                    error = ModuleNotFoundError("No module named 'mcp'")
                    error.name = "mcp"
                    raise error
                return original_import(name, *args, **kwargs)

            builtins.__import__ = blocked_import
            try:
                import raglight.agentic.mcp
            except ImportError as error:
                message = str(error)
                raise SystemExit(0 if "raglight[agentic-mcp]" in message else message)
            raise SystemExit("MCP import unexpectedly succeeded")
            """)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == "__main__":
    unittest.main()
