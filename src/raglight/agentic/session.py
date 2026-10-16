"""Conversation state isolated to one caller of an agentic pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Protocol
from uuid import uuid4

from ..models import RAGResponse


class _AgenticSessionPipeline(Protocol):
    def generate(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> RAGResponse: ...


class AgenticSession:
    """Own completed conversation turns for one agentic caller."""

    def __init__(
        self,
        pipeline: _AgenticSessionPipeline,
        *,
        session_id: str | None = None,
        history: Sequence[Mapping[str, str]] = (),
    ) -> None:
        normalized_id = str(uuid4()) if session_id is None else session_id.strip()
        if not normalized_id:
            raise ValueError("session_id must not be empty")
        self._pipeline = pipeline
        self._session_id = normalized_id
        self._history = [self._normalize_message(message) for message in history]
        self._lock = RLock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        with self._lock:
            return tuple(dict(message) for message in self._history)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def generate(self, question: str) -> RAGResponse:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        with self._lock:
            response = self._pipeline.generate(normalized, history=self._history)
            self._history.extend(
                (
                    {"role": "user", "content": normalized},
                    {"role": "assistant", "content": response.answer},
                )
            )
            return response

    @staticmethod
    def _normalize_message(message: Mapping[str, str]) -> dict[str, str]:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("history message role must not be empty")
        if not isinstance(content, str):
            raise ValueError("history message content must be a string")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in message.items()
        ):
            raise ValueError("history message keys and values must be strings")
        value = dict(message)
        value["role"] = role.strip()
        return value
