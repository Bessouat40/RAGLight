"""Unit tests for isolated conversation sessions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import unittest
from uuid import UUID

from raglight.models import RAGResponse
from raglight.pipeline import ConversationSession


class RecordingPipeline:
    def __init__(self) -> None:
        self.generate_calls = []
        self.stream_calls = []
        self.generate_error: Exception | None = None
        self.stream_error: Exception | None = None

    def generate(self, question, *, history=()):
        self.generate_calls.append((question, [dict(message) for message in history]))
        if self.generate_error is not None:
            raise self.generate_error
        return RAGResponse(answer=f"answer:{question}", trace_id="trace")

    def stream(self, question, *, history=()):
        self.stream_calls.append((question, [dict(message) for message in history]))
        yield "answer:"
        if self.stream_error is not None:
            raise self.stream_error
        yield question


class ConcurrentPipeline:
    def __init__(self) -> None:
        self.barrier = Barrier(2)
        self.histories = {}
        self.lock = Lock()

    def generate(self, question, *, history=()):
        self.barrier.wait(timeout=5)
        with self.lock:
            self.histories[question] = [dict(message) for message in history]
        return RAGResponse(answer=f"answer:{question}", trace_id=question)

    def stream(self, question, *, history=()):
        yield question


class ConversationSessionTests(unittest.TestCase):
    def test_default_session_id_is_a_uuid(self):
        session = ConversationSession(RecordingPipeline())

        self.assertEqual(session.session_id, str(UUID(session.session_id)))

    def test_session_copies_initial_history_and_exposes_snapshots(self):
        pipeline = RecordingPipeline()
        initial_history = [{"role": "user", "content": "Earlier"}]
        session = ConversationSession(
            pipeline,
            session_id="conversation-1",
            history=initial_history,
        )

        initial_history[0]["content"] = "Changed outside"
        snapshot = session.history
        snapshot[0]["content"] = "Changed snapshot"

        self.assertEqual("conversation-1", session.session_id)
        self.assertEqual(
            ({"role": "user", "content": "Earlier"},),
            session.history,
        )

    def test_generate_forwards_history_and_commits_completed_turn(self):
        pipeline = RecordingPipeline()
        session = ConversationSession(pipeline, session_id="conversation-1")

        response = session.generate("First question")
        session.generate("Follow-up")

        self.assertEqual("answer:First question", response.answer)
        self.assertEqual(
            [
                ("First question", []),
                (
                    "Follow-up",
                    [
                        {"role": "user", "content": "First question"},
                        {
                            "role": "assistant",
                            "content": "answer:First question",
                        },
                    ],
                ),
            ],
            pipeline.generate_calls,
        )
        self.assertEqual(
            (
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "answer:First question"},
                {"role": "user", "content": "Follow-up"},
                {"role": "assistant", "content": "answer:Follow-up"},
            ),
            session.history,
        )

    def test_concurrent_sessions_do_not_share_history_or_locks(self):
        pipeline = ConcurrentPipeline()
        first = ConversationSession(
            pipeline,
            session_id="first",
            history=[{"role": "user", "content": "First earlier message"}],
        )
        second = ConversationSession(
            pipeline,
            session_id="second",
            history=[{"role": "user", "content": "Second earlier message"}],
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda item: item[0].generate(item[1]),
                    ((first, "First question"), (second, "Second question")),
                )
            )

        self.assertEqual(
            ["answer:First question", "answer:Second question"],
            [response.answer for response in results],
        )
        self.assertEqual(
            [{"role": "user", "content": "First earlier message"}],
            pipeline.histories["First question"],
        )
        self.assertEqual(
            [{"role": "user", "content": "Second earlier message"}],
            pipeline.histories["Second question"],
        )

    def test_failed_generation_does_not_change_history(self):
        pipeline = RecordingPipeline()
        pipeline.generate_error = RuntimeError("generation failed")
        session = ConversationSession(pipeline, session_id="conversation-1")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            session.generate("Question")

        self.assertEqual((), session.history)

    def test_stream_commits_only_after_complete_consumption(self):
        pipeline = RecordingPipeline()
        session = ConversationSession(pipeline, session_id="conversation-1")

        stream = iter(session.stream("Question"))
        self.assertEqual("answer:", next(stream))
        self.assertEqual((), session.history)
        self.assertEqual("Question", next(stream))
        with self.assertRaises(StopIteration):
            next(stream)

        self.assertEqual(
            (
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "answer:Question"},
            ),
            session.history,
        )
        self.assertEqual([("Question", [])], pipeline.stream_calls)

    def test_abandoned_or_failed_stream_does_not_change_history(self):
        pipeline = RecordingPipeline()
        session = ConversationSession(pipeline, session_id="conversation-1")

        abandoned = iter(session.stream("Abandoned"))
        next(abandoned)
        abandoned.close()
        self.assertEqual((), session.history)

        pipeline.stream_error = RuntimeError("stream failed")
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            list(session.stream("Failed"))
        self.assertEqual((), session.history)

    def test_clear_removes_history_without_changing_identity(self):
        session = ConversationSession(
            RecordingPipeline(),
            session_id="conversation-1",
            history=[{"role": "user", "content": "Earlier"}],
        )

        session.clear()

        self.assertEqual("conversation-1", session.session_id)
        self.assertEqual((), session.history)

    def test_invalid_session_inputs_are_rejected(self):
        pipeline = RecordingPipeline()

        with self.assertRaisesRegex(ValueError, "session_id must not be empty"):
            ConversationSession(pipeline, session_id=" ")
        with self.assertRaisesRegex(ValueError, "role must not be empty"):
            ConversationSession(
                pipeline,
                history=[{"role": " ", "content": "message"}],
            )
        with self.assertRaisesRegex(ValueError, "content must be a string"):
            ConversationSession(
                pipeline,
                history=[{"role": "user", "content": 3}],
            )


if __name__ == "__main__":
    unittest.main()
