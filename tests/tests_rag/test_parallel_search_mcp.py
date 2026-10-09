import json
import os
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.parallel_search_mcp import create_config
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.rag.agentic_rag import AgenticRAG
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline


class SearchThenFetchModel(BaseChatModel):
    """Deterministic model that exercises the real LangGraph tool loop."""

    @property
    def _llm_type(self):
        return "test-search-then-fetch"

    def bind_tools(self, tools, **kwargs):
        names = {tool.name for tool in tools}
        if "web_search" in names or "web_fetch" in names:
            assert {"web_search", "web_fetch"}.issubset(names)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [message for message in messages if isinstance(message, ToolMessage)]
        if not results:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search",
                        "name": "web_search",
                        "args": {
                            "objective": "Find Python docs",
                            "search_queries": ["Python docs"],
                        },
                    }
                ],
            )
        elif len(results) == 1:
            assert "https://docs.python.org/3/" in str(results[0].content), results[
                0
            ].content
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "fetch",
                        "name": "web_fetch",
                        "args": {"urls": ["https://docs.python.org/3/"]},
                    }
                ],
            )
        else:
            assert "Python documentation" in str(results[-1].content), results[
                -1
            ].content
            message = AIMessage(content=str(results[-1].content))
        return ChatResult(generations=[ChatGeneration(message=message)])


class TestParallelSearchMCP(unittest.TestCase):
    def test_pipeline_dispatches_search_and_fetch_without_credentials(self):
        # tests/__init__.py globally stubs LangGraph ToolNode. Run this test
        # in a fresh interpreter so the actual agent loop is exercised.
        if __name__ != "__main__":
            root = Path(__file__).resolve().parents[2]
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "TestParallelSearchMCP.test_pipeline_dispatches_search_and_fetch_without_credentials",
                ],
                cwd=root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]),
                },
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        requests = []
        schemas = {
            "web_search": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["objective", "search_queries"],
            },
            "web_fetch": {
                "type": "object",
                "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
                "required": ["urls"],
            },
        }

        def handle(request):
            self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            self.assertEqual(
                request.headers["User-Agent"],
                "RAGLight-Parallel-MCP/1.0 (https://github.com/Bessouat40/RAGLight)",
            )
            self.assertNotIn("authorization", request.headers)
            if request.method != "POST":
                return httpx.Response(405)
            body = json.loads(request.content)
            requests.append(body)
            if "id" not in body:
                return httpx.Response(202)
            if body["method"] == "initialize":
                result = {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-parallel", "version": "1"},
                }
            elif body["method"] == "tools/list":
                result = {
                    "tools": [
                        {"name": name, "description": name, "inputSchema": schema}
                        for name, schema in schemas.items()
                    ]
                }
            elif body["method"] == "tools/call":
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Python documentation: https://docs.python.org/3/"
                            ),
                        }
                    ],
                    "isError": False,
                }
            else:
                self.fail(f"Unexpected MCP method: {body['method']}")
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}
            )

        def client_factory(headers=None, timeout=None, auth=None):
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                auth=auth,
                transport=httpx.MockTransport(handle),
            )

        config = create_config()
        config.mcp_config["parallel"]["httpx_client_factory"] = client_factory
        # Only external model/vector infrastructure is replaced. MCP loading,
        # transport, tool conversion and agent dispatch use the actual callers.
        with (
            patch.object(AgenticRAG, "create_vector_store", return_value=MagicMock()),
            patch.object(
                AgenticRAG, "_create_llm_model", return_value=SearchThenFetchModel()
            ),
            patch.dict("os.environ", {"PARALLEL_API_KEY": "must-not-be-sent"}),
        ):
            with AgenticRAGPipeline(
                config, VectorStoreConfig(embedding_model="test")
            ) as pipeline:
                pipeline.build()
                answer = pipeline.generate("Find and read Python documentation")
                self.assertIn("Python documentation", answer)
                self.assertIn("https://docs.python.org/3/", answer)
        calls = [r["params"]["name"] for r in requests if r["method"] == "tools/call"]
        self.assertEqual(calls, ["web_search", "web_fetch"])
        self.assertTrue(any(r["method"] == "tools/list" for r in requests))

    def test_defaults_and_connections_are_independent(self):
        self.assertIsNone(AgenticRAGConfig().mcp_config)
        first, second = create_config(), create_config()
        first.mcp_config["parallel"]["headers"]["User-Agent"] = "changed"
        self.assertNotEqual(first.mcp_config, second.mcp_config)
        self.assertEqual(second.provider, AgenticRAGConfig().provider)


if __name__ == "__main__":
    unittest.main()
