import unittest
from unittest.mock import MagicMock, patch

from raglight.integrations.chroma import ChromaVectorStore
from raglight.models import Document
from raglight.integrations.vectorstores.bm25_index import BM25Index


def _make_chroma(search_type: str = "semantic") -> ChromaVectorStore:
    """Return a Chroma vector store with all external dependencies mocked."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2]]

    with patch("raglight.integrations.vectorstores.chroma.chromadb") as mock_chromadb:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_col"
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        vs = ChromaVectorStore(
            collection_name="test_col",
            embeddings_model=mock_embeddings,
            persist_directory="/tmp/test_chroma",
            search_type=search_type,
        )
    return vs


class TestBM25Index(unittest.TestCase):
    def test_add_and_search(self):
        index = BM25Index()
        index.add_documents(["the quick brown fox", "lazy dog over fence"])
        results = index.search("quick fox", k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        idx, score = results[0]
        # First result must be the fox document (index 0) since "quick" and "fox" only appear there
        self.assertEqual(idx, 0)

    def test_empty_search(self):
        index = BM25Index()
        results = index.search("query", k=5)
        self.assertEqual(results, [])

    def test_save_and_load(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        index = BM25Index()
        index.add_documents(["hello world", "foo bar"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bm25.json"
            index.save(path)
            index2 = BM25Index()
            index2.load(path)
            self.assertEqual(index2.corpus, ["hello world", "foo bar"])
            results = index2.search("hello", k=1)
            self.assertEqual(len(results), 1)


class TestRRFFusion(unittest.TestCase):
    def test_rrf_deduplicates_and_ranks(self):
        vs = _make_chroma("semantic")
        doc_a = Document(page_content="document about python programming")
        doc_b = Document(page_content="document about machine learning")
        doc_c = Document(page_content="something completely different here")

        # doc_a appears in both lists (rank 0 each) → highest RRF score
        result = vs._rrf([[doc_a, doc_b], [doc_a, doc_c]])
        self.assertEqual(result[0].page_content, doc_a.page_content)
        # deduplication: doc_a appears only once
        contents = [d.page_content for d in result]
        self.assertEqual(len(contents), len(set(contents)))

    def test_rrf_combines_all_docs(self):
        vs = _make_chroma("semantic")
        doc_a = Document(page_content="alpha")
        doc_b = Document(page_content="beta")
        doc_c = Document(page_content="gamma")

        result = vs._rrf([[doc_a], [doc_b], [doc_c]])
        self.assertEqual(len(result), 3)


class TestBM25SearchMode(unittest.TestCase):
    def test_bm25_mode_returns_documents(self):
        vs = _make_chroma("bm25")
        vs._bm25.add_documents(["cat sat on the mat", "dog ran in the park"])
        results = vs.similarity_search("cat mat", k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)

    def test_bm25_mode_does_not_call_chroma(self):
        vs = _make_chroma("bm25")
        vs._bm25.add_documents(["some text here"])
        vs._query_collection = MagicMock()
        vs.similarity_search("some text", k=1)
        vs._query_collection.assert_not_called()


class TestHybridSearchMode(unittest.TestCase):
    def test_hybrid_calls_both_semantic_and_bm25(self):
        vs = _make_chroma("hybrid")
        vs._bm25.add_documents(["cat sat on the mat", "dog ran in the park"])

        semantic_docs = [Document(page_content="cat sat on the mat")]
        vs._query_collection = MagicMock(return_value=semantic_docs)

        results = vs.similarity_search("cat", k=1)
        vs._query_collection.assert_called_once()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_hybrid_result_is_deduplicated(self):
        vs = _make_chroma("hybrid")
        vs._bm25.add_documents(["shared document content here"])

        shared_doc = Document(page_content="shared document content here")
        vs._query_collection = MagicMock(return_value=[shared_doc])

        results = vs.similarity_search("shared", k=5)
        contents = [d.page_content for d in results]
        self.assertEqual(len(contents), len(set(contents)))


class TestSemanticModeUnchanged(unittest.TestCase):
    def test_semantic_mode_delegates_to_query_collection(self):
        vs = _make_chroma("semantic")
        expected = [Document(page_content="result doc")]
        vs._query_collection = MagicMock(return_value=expected)

        results = vs.similarity_search("query", k=3)
        vs._query_collection.assert_called_once()
        self.assertEqual(results, expected)

    def test_semantic_mode_does_not_call_bm25(self):
        vs = _make_chroma("semantic")
        vs._bm25.search = MagicMock(return_value=[])
        vs._query_collection = MagicMock(return_value=[])

        vs.similarity_search("query", k=3)
        vs._bm25.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
