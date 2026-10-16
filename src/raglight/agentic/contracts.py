"""Structural extension contracts owned by the agentic capability."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Any, Protocol, runtime_checkable

from .models import AgentContext, AgentDecision, ToolResult


@runtime_checkable
class Agent(Protocol):
    """Choose the next bounded action from an immutable context snapshot."""

    def decide(
        self, context: AgentContext
    ) -> AgentDecision | Awaitable[AgentDecision]: ...


@runtime_checkable
class AgentTool(Protocol):
    """Execute one named operation selected by an agent."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def run(
        self, arguments: Mapping[str, Any]
    ) -> ToolResult | Awaitable[ToolResult]: ...
