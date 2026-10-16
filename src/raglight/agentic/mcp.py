"""Optional adapter from an initialized MCP client to an agent tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

try:
    from mcp import types as mcp_types
except ModuleNotFoundError as error:
    from .._optional import optional_import_error

    optional_import_error(
        error,
        dependency="mcp",
        extra="agentic-mcp",
        feature="Agentic MCP tool support",
    )

from .models import ToolResult, ToolSpec


class MCPClient(Protocol):
    """Minimal initialized-client surface required by ``MCPTool``."""

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> Any: ...


class MCPToolError(RuntimeError):
    """Raised when an MCP server reports a failed tool invocation."""


class MCPTool:
    """Adapt one tool on a caller-managed MCP client connection.

    The caller owns the MCP client's connection lifecycle. Use this tool with
    ``await AgenticPipeline.agenerate(...)`` while that client is connected.
    """

    def __init__(
        self,
        client: MCPClient,
        name: str,
        description: str,
        input_schema: Mapping[str, Any] | None = None,
    ) -> None:
        spec = ToolSpec(name, description, input_schema or {})
        self._client = client
        self.name = spec.name
        self.description = spec.description
        self.input_schema = spec.input_schema

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        result = await self._client.call_tool(self.name, dict(arguments))
        text = self._text_content(result)
        if bool(getattr(result, "is_error", getattr(result, "isError", False))):
            detail = text or "the MCP server returned an unspecified error"
            raise MCPToolError(f"MCP tool '{self.name}' failed: {detail}")

        structured = getattr(
            result,
            "structured_content",
            getattr(result, "structuredContent", None),
        )
        metadata: dict[str, Any] = {"mcp": True}
        if structured is not None:
            metadata["structured_content"] = structured
        if not text and structured is not None:
            text = str(structured)
        return ToolResult(
            text or f"MCP tool '{self.name}' completed without text content.",
            metadata=metadata,
        )

    @staticmethod
    def _text_content(result: Any) -> str:
        blocks = getattr(result, "content", ())
        return "\n".join(
            block.text for block in blocks if isinstance(block, mcp_types.TextContent)
        )


__all__ = ["MCPClient", "MCPTool", "MCPToolError"]
