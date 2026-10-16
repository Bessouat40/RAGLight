"""Default JSON decision planner backed by the public LLM contract."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from ..contracts import LLM
from .models import AgentContext, AgentDecision


class AgentDecisionError(ValueError):
    """Raised when an agent does not return a valid bounded decision."""


class LLMPlannerAgent:
    """Ask an injected v4 LLM for one strict JSON decision per step."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def decide(self, context: AgentContext) -> AgentDecision:
        context_value = context.to_dict()
        prompt = self._build_prompt(context_value)
        response = self._llm.generate(
            {
                "mode": "agentic_decision",
                "question": prompt,
                "original_question": context.question,
                "history": context_value["history"],
                "tools": context_value["tools"],
                "steps": context_value["steps"],
                "instructions": (
                    "Return one JSON object only. Use "
                    '{"type":"tool","tool_name":"name","arguments":{...}} '
                    "to call a tool, or "
                    '{"type":"final","answer":"..."} to finish.'
                ),
            }
        )
        try:
            value = self._parse_json_object(response)
        except (json.JSONDecodeError, TypeError) as error:
            raise AgentDecisionError(
                "agent output must be a valid JSON decision object"
            ) from error
        if not isinstance(value, Mapping):
            raise AgentDecisionError(
                "agent output must be a valid JSON decision object"
            )
        try:
            decision_type = value.get("type")
            if decision_type == "tool":
                arguments = value.get("arguments", {})
                if not isinstance(arguments, Mapping):
                    raise TypeError("tool arguments must be a mapping")
                return AgentDecision.call_tool(
                    str(value.get("tool_name", "")),
                    arguments,
                )
            if decision_type == "final":
                return AgentDecision.final(str(value.get("answer", "")))
            raise ValueError(f"unsupported agent decision: {decision_type}")
        except (TypeError, ValueError) as error:
            raise AgentDecisionError(str(error)) from error

    @staticmethod
    def _strip_fence(value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @classmethod
    def _parse_json_object(cls, value: str) -> Mapping[str, Any]:
        stripped = cls._strip_fence(value)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, character in enumerate(stripped):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(stripped[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, Mapping):
                    return candidate
            raise
        if not isinstance(parsed, Mapping):
            raise TypeError("agent output must be a JSON object")
        return parsed

    @staticmethod
    def _build_prompt(context: Mapping[str, Any]) -> str:
        return (
            "You are the decision planner in a bounded agentic RAG loop. "
            "Choose exactly one next action.\n\n"
            f"Original question:\n{context['question']}\n\n"
            "Available tools:\n"
            f"{json.dumps(context['tools'], ensure_ascii=False, default=str)}\n\n"
            "Previous steps and observations:\n"
            f"{json.dumps(context['steps'], ensure_ascii=False, default=str)}\n\n"
            "Return one JSON object only. Use "
            '{"type":"tool","tool_name":"name","arguments":{...}} '
            "to call a tool, or "
            '{"type":"final","answer":"..."} to finish.'
        )
