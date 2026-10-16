"""Shared scoped-retrieval behavior for built-in semantic stores."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.qdrant import QdrantVectorStore
from raglight.models import And, Document, Eq, In, MatchNone, Or


class ConstantEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class ScopedVectorStoreIntegrationTests(unittest.TestCase):
    def test_qdrant_and_chroma_apply_the_same_filter_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stores = (
                ChromaVectorStore(
                    collection_name="scoped_chroma",
                    embeddings_model=ConstantEmbeddings(),
                    persist_directory=str(root / "chroma"),
                ),
                QdrantVectorStore(
                    collection_name="scoped_qdrant",
                    embeddings_model=ConstantEmbeddings(),
                    persist_directory=str(root / "qdrant"),
                ),
            )
            documents = [
                Document(
                    "Alice private",
                    {
                        "workspace_id": "one",
                        "owner": "alice",
                        "score": 1.5,
                        "shared": False,
                    },
                ),
                Document(
                    "Bob private",
                    {
                        "workspace_id": "one",
                        "owner": "bob",
                        "score": 2.0,
                        "shared": True,
                    },
                ),
                Document(
                    "Alice other",
                    {
                        "workspace_id": "two",
                        "owner": "alice",
                        "score": 3.0,
                        "shared": False,
                    },
                ),
            ]
            filters = (
                (Eq("owner", "bob"), {"Bob private"}),
                (In("owner", ["bob"]), {"Bob private"}),
                (
                    And(Eq("workspace_id", "one"), Eq("owner", "alice")),
                    {"Alice private"},
                ),
                (
                    Or(Eq("owner", "bob"), Eq("workspace_id", "two")),
                    {"Bob private", "Alice other"},
                ),
                (Eq("score", 1.5), {"Alice private"}),
                (In("score", [1.5, 3.0]), {"Alice private", "Alice other"}),
                (In("shared", [True]), {"Bob private"}),
                (MatchNone(), set()),
            )

            try:
                for store in stores:
                    store.add_documents(documents)
                    for retrieval_filter, expected in filters:
                        with self.subTest(
                            store=type(store).__name__,
                            retrieval_filter=retrieval_filter,
                        ):
                            results = store.similarity_search(
                                "private",
                                k=10,
                                retrieval_filter=retrieval_filter,
                            )
                            self.assertEqual(
                                expected,
                                {document.page_content for document in results},
                            )
            finally:
                stores[1].client.close()


if __name__ == "__main__":
    unittest.main()
