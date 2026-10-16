"""Unit tests for provider-neutral retrieval filters."""

from __future__ import annotations

import unittest

from raglight.models import And, Eq, In, MatchNone, Or


class RetrievalFilterModelTests(unittest.TestCase):
    def test_composite_filters_are_immutable_and_preserve_typed_children(self):
        file_ids = ["file-a", "file-b"]
        scope = And(
            Eq("workspace_id", "ws-123"),
            Or(In("file_id", file_ids), MatchNone()),
        )

        file_ids.append("file-c")

        self.assertEqual(("file-a", "file-b"), scope.filters[1].filters[0].values)
        with self.assertRaisesRegex(AttributeError, "cannot assign"):
            scope.filters = ()

    def test_fields_and_values_are_validated_at_the_public_boundary(self):
        invalid = (
            (ValueError, "field must not be empty", lambda: Eq(" ", "value")),
            (TypeError, "scalar", lambda: Eq("field", {"nested": "value"})),
            (ValueError, "at least one value", lambda: In("file_id", [])),
            (ValueError, "at least one filter", lambda: And()),
            (TypeError, "retrieval filters", lambda: Or(Eq("a", 1), "invalid")),
        )

        for error_type, message, build in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    build()

    def test_pipeline_filter_normalization_rejects_provider_dictionaries(self):
        from raglight.models.retrieval_filter import normalize_retrieval_filter

        with self.assertRaisesRegex(TypeError, "RAGLight retrieval filter"):
            normalize_retrieval_filter({"workspace_id": "ws-1"})


if __name__ == "__main__":
    unittest.main()
