"""Unit tests for provider-independent structured response models."""

from __future__ import annotations

import unittest

from raglight.models import Document, ExecutionEvent, RAGResponse


class ExecutionEventTests(unittest.TestCase):
    def test_event_values_are_immutable_and_serializable(self):
        event = ExecutionEvent(
            stage="retrieval",
            status="completed",
            duration_ms=1.25,
            metrics={"document_count": 2},
            metadata={"collection": "docs"},
        )

        with self.assertRaises(TypeError):
            event.metrics["document_count"] = 3
        with self.assertRaises(TypeError):
            event.metadata["collection"] = "other"

        self.assertEqual(
            {
                "stage": "retrieval",
                "status": "completed",
                "duration_ms": 1.25,
                "metrics": {"document_count": 2},
                "message": None,
                "metadata": {"collection": "docs"},
            },
            event.to_dict(),
        )

    def test_event_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "stage must not be empty"):
            ExecutionEvent(stage=" ", status="completed", duration_ms=0)
        with self.assertRaisesRegex(ValueError, "unsupported event status"):
            ExecutionEvent(stage="retrieval", status="unknown", duration_ms=0)
        with self.assertRaisesRegex(ValueError, "duration_ms must not be negative"):
            ExecutionEvent(stage="retrieval", status="completed", duration_ms=-1)


class RAGResponseTests(unittest.TestCase):
    def test_response_normalizes_collections_and_serializes(self):
        source = Document(page_content="context", metadata={"page": 4})
        event = ExecutionEvent(
            stage="generation",
            status="completed",
            duration_ms=2,
        )
        response = RAGResponse(
            answer="answer",
            sources=[source],
            events=[event],
            timings={"generation": 2.0, "total": 2.5},
            metadata={"top_k": 5},
            trace_id="trace-123",
        )

        self.assertEqual((source,), response.sources)
        self.assertEqual((event,), response.events)
        with self.assertRaises(TypeError):
            response.timings["total"] = 3
        with self.assertRaises(TypeError):
            response.metadata["top_k"] = 10

        self.assertEqual(
            {
                "answer": "answer",
                "sources": [{"page_content": "context", "metadata": {"page": 4}}],
                "events": [event.to_dict()],
                "timings": {"generation": 2.0, "total": 2.5},
                "metadata": {"top_k": 5},
                "trace_id": "trace-123",
            },
            response.to_dict(),
        )

    def test_response_requires_a_trace_id_and_nonnegative_timings(self):
        with self.assertRaisesRegex(ValueError, "trace_id must not be empty"):
            RAGResponse(answer="answer", trace_id=" ")
        with self.assertRaisesRegex(ValueError, "timings must not be negative"):
            RAGResponse(
                answer="answer",
                trace_id="trace-123",
                timings={"total": -0.1},
            )


if __name__ == "__main__":
    unittest.main()
