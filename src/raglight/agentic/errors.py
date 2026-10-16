"""Typed errors raised by bounded agentic execution."""

from __future__ import annotations

from collections.abc import Sequence

from ..models import ExecutionEvent


class AgenticExecutionError(RuntimeError):
    """Base error carrying safe execution details for delivery layers."""

    def __init__(
        self,
        message: str,
        *,
        trace_id: str,
        events: Sequence[ExecutionEvent],
    ) -> None:
        super().__init__(message)
        self.trace_id = trace_id
        self.events = tuple(events)


class AgentStepLimitError(AgenticExecutionError):
    """Raised when no final answer is produced within the configured budget."""

    def __init__(
        self,
        *,
        max_steps: int,
        trace_id: str,
        events: Sequence[ExecutionEvent],
    ) -> None:
        super().__init__(
            f"Agent reached the maximum of {max_steps} step(s) without a final answer.",
            trace_id=trace_id,
            events=events,
        )
        self.max_steps = max_steps
