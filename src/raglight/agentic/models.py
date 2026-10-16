"""Provider-independent values used by the agentic runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ..models import Document

AgentDecisionType = Literal["tool", "final"]


def _mapping_snapshot(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Describe one tool without exposing its implementation to an agent."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        description = self.description.strip()
        if not name:
            raise ValueError("tool name must not be empty")
        if not description:
            raise ValueError("tool description must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(
            self,
            "input_schema",
            _mapping_snapshot(self.input_schema, name="input_schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Return model-readable content and optional RAG source documents."""

    content: str
    sources: Sequence[Document] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("tool result content must be a string")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(
            self,
            "metadata",
            _mapping_snapshot(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "sources": [
                {
                    "page_content": source.page_content,
                    "metadata": dict(source.metadata),
                }
                for source in self.sources
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """Ask the runtime to execute a tool or return a final answer."""

    type: AgentDecisionType | str
    tool_name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    answer: str | None = None

    def __post_init__(self) -> None:
        decision_type = self.type.strip().lower()
        if decision_type not in {"tool", "final"}:
            raise ValueError(f"unsupported agent decision: {self.type}")
        object.__setattr__(self, "type", decision_type)
        object.__setattr__(
            self,
            "arguments",
            _mapping_snapshot(self.arguments, name="arguments"),
        )

        if decision_type == "tool":
            name = "" if self.tool_name is None else self.tool_name.strip().lower()
            if not name:
                raise ValueError("tool name must not be empty")
            if self.answer is not None:
                raise ValueError("a tool decision must not contain a final answer")
            object.__setattr__(self, "tool_name", name)
            return

        answer = "" if self.answer is None else self.answer.strip()
        if not answer:
            raise ValueError("final answer must not be empty")
        if self.tool_name is not None:
            raise ValueError("a final decision must not contain a tool name")
        if self.arguments:
            raise ValueError("a final decision must not contain tool arguments")
        object.__setattr__(self, "answer", answer)

    @classmethod
    def call_tool(
        cls,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> AgentDecision:
        return cls(type="tool", tool_name=name, arguments=arguments or {})

    @classmethod
    def final(cls, answer: str) -> AgentDecision:
        return cls(type="final", answer=answer)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"type": self.type}
        if self.type == "tool":
            value.update(
                {"tool_name": self.tool_name, "arguments": dict(self.arguments)}
            )
        else:
            value["answer"] = self.answer
        return value


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """Record the outcome made visible to the next agent decision."""

    tool_name: str
    content: str
    sources: Sequence[Document] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    is_error: bool = False

    def __post_init__(self) -> None:
        name = self.tool_name.strip().lower()
        if not name:
            raise ValueError("tool name must not be empty")
        object.__setattr__(self, "tool_name", name)
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(
            self,
            "metadata",
            _mapping_snapshot(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "content": self.content,
            "sources": [
                {
                    "page_content": source.page_content,
                    "metadata": dict(source.metadata),
                }
                for source in self.sources
            ],
            "metadata": dict(self.metadata),
            "is_error": self.is_error,
        }


@dataclass(frozen=True, slots=True)
class AgentStep:
    """Pair one agent decision with its optional tool observation."""

    number: int
    decision: AgentDecision
    observation: ToolObservation | None = None

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("agent step number must be greater than zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "decision": self.decision.to_dict(),
            "observation": (
                None if self.observation is None else self.observation.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Provide one immutable decision snapshot to an agent."""

    question: str
    history: Sequence[Mapping[str, str]] = ()
    tools: Sequence[ToolSpec] = ()
    steps: Sequence[AgentStep] = ()

    def __post_init__(self) -> None:
        question = self.question.strip()
        if not question:
            raise ValueError("question must not be empty")
        history = tuple(MappingProxyType(dict(message)) for message in self.history)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "steps", tuple(self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "history": [dict(message) for message in self.history],
            "tools": [tool.to_dict() for tool in self.tools],
            "steps": [step.to_dict() for step in self.steps],
        }
