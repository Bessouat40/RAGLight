"""Conversation state scoped to one caller of a shared RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from threading import RLock
from typing import Protocol
from uuid import uuid4

from ..models import RAGResponse


class _SessionPipeline(Protocol):
    def generate(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> RAGResponse: ...

    def stream(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
    ) -> Iterable[str]: ...


class ConversationSession:
    """Own the isolated history for one conversation.

    Sessions may share a pipeline and its configured components. Each session
    serializes its own operations so a turn always observes a complete history.
    Failed or partially consumed turns are not committed.
    """

    def __init__(
        self,
        pipeline: _SessionPipeline,
        *,
        session_id: str | None = None,
        history: Sequence[Mapping[str, str]] = (),
    ) -> None:
        normalized_session_id = (
            str(uuid4()) if session_id is None else session_id.strip()
        )
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")

        self._pipeline = pipeline
        self._session_id = normalized_session_id
        self._history = [self._normalize_message(message) for message in history]
        self._lock = RLock()

    @property
    def session_id(self) -> str:
        """Return the stable identifier assigned to this conversation."""

        return self._session_id

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        """Return a detached snapshot of the committed conversation history."""

        with self._lock:
            return tuple(dict(message) for message in self._history)

    def clear(self) -> None:
        """Remove all committed turns while preserving the session identity."""

        with self._lock:
            self._history.clear()

    def generate(self, question: str) -> RAGResponse:
        """Generate an answer and commit the completed turn to this session."""

        normalized_question = self._normalize_question(question)
        with self._lock:
            response = self._pipeline.generate(
                normalized_question,
                history=self._history,
            )
            self._commit(normalized_question, response.answer)
            return response

    def stream(self, question: str) -> Iterable[str]:
        """Yield answer chunks and commit only after the stream completes."""

        normalized_question = self._normalize_question(question)

        def completed_stream() -> Iterable[str]:
            with self._lock:
                chunks: list[str] = []
                for chunk in self._pipeline.stream(
                    normalized_question,
                    history=self._history,
                ):
                    chunks.append(chunk)
                    yield chunk
                self._commit(normalized_question, "".join(chunks))

        return completed_stream()

    def _commit(self, question: str, answer: str) -> None:
        self._history.extend(
            (
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            )
        )

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized

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
        normalized = dict(message)
        normalized["role"] = role.strip()
        return normalized
