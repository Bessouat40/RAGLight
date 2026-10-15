import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from raglight.rag.rag import RAG


class PipelineIntegrationTests(unittest.TestCase):
    def test_retrieved_context_reaches_generation(self):
        embeddings = MagicMock()
        embeddings.get_model.return_value = MagicMock()
        vector_store = MagicMock()
        vector_store.similarity_search.return_value = [
            Document(page_content="RAGLight keeps retrieval modular.")
        ]
        llm = MagicMock()
        llm.generate.return_value = "Retrieval is modular."
        pipeline = RAG(
            embedding_model=embeddings,
            vector_store=vector_store,
            llm=llm,
            k=1,
            reformulation=False,
        )

        answer = pipeline.generate("How does retrieval work?")

        self.assertEqual("Retrieval is modular.", answer)
        vector_store.similarity_search.assert_called_once_with(
            "How does retrieval work?", k=1
        )
        generation_input = llm.generate.call_args.args[0]
        self.assertIn("RAGLight keeps retrieval modular.", generation_input["question"])


if __name__ == "__main__":
    unittest.main()
