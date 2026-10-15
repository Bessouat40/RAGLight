import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from raglight import RAG


class PublicRAGWorkflowTests(unittest.TestCase):
    def test_question_to_answer_workflow_uses_public_import(self):
        embeddings = MagicMock()
        embeddings.get_model.return_value = MagicMock()
        vector_store = MagicMock()
        vector_store.similarity_search.return_value = [
            Document(page_content="RAGLight is a modular RAG library.")
        ]
        llm = MagicMock()
        llm.generate.return_value = "RAGLight is modular."
        pipeline = RAG(
            embedding_model=embeddings,
            vector_store=vector_store,
            llm=llm,
            k=1,
            reformulation=False,
        )

        answer = pipeline.generate("What is RAGLight?")

        self.assertEqual("RAGLight is modular.", answer)
        self.assertEqual(
            [
                {"role": "user", "content": "What is RAGLight?"},
                {"role": "assistant", "content": "RAGLight is modular."},
            ],
            pipeline.state["history"],
        )


if __name__ == "__main__":
    unittest.main()
