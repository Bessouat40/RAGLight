"""Provider translation tests for the shared retrieval-filter AST."""

from __future__ import annotations

import unittest

from raglight.integrations.vectorstores.chroma import _to_chroma_filter
from raglight.integrations.vectorstores.qdrant import _to_qdrant_filter
from raglight.models import And, Eq, In, Or


class RetrievalFilterTranslationTests(unittest.TestCase):
    def setUp(self):
        self.scope = And(
            Eq("workspace_id", "ws-1"),
            Or(
                In("file_id", ["file-a", "file-b"]),
                Eq("shared", True),
            ),
        )

    def test_chroma_translates_the_provider_neutral_filter(self):
        self.assertEqual(
            {
                "$and": [
                    {"workspace_id": {"$eq": "ws-1"}},
                    {
                        "$or": [
                            {"file_id": {"$in": ["file-a", "file-b"]}},
                            {"shared": {"$eq": True}},
                        ]
                    },
                ]
            },
            _to_chroma_filter(self.scope),
        )

    def test_qdrant_translates_the_same_logical_tree(self):
        translated = _to_qdrant_filter(self.scope)

        self.assertEqual("ws-1", translated.must[0].match.value)
        self.assertEqual(["file-a", "file-b"], translated.must[1].should[0].match.any)
        self.assertIs(True, translated.must[1].should[1].match.value)

    def test_qdrant_preserves_float_equality_and_mixed_in_values(self):
        equality = _to_qdrant_filter(Eq("score", 1.5))
        alternatives = _to_qdrant_filter(In("score", [1, 1.5]))
        booleans = _to_qdrant_filter(In("shared", [True]))

        self.assertEqual(1.5, equality.must[0].range.gte)
        self.assertEqual(1.5, equality.must[0].range.lte)
        self.assertEqual(1, alternatives.must[0].should[0].match.value)
        self.assertEqual(1.5, alternatives.must[0].should[1].range.gte)
        self.assertIs(True, booleans.must[0].should[0].match.value)


if __name__ == "__main__":
    unittest.main()
