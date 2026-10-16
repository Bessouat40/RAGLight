"""Tests for the public, structural extension contracts."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
import unittest

import raglight.contracts as contracts
from raglight.contracts import (
    DocumentProcessor,
    EmbeddingModel,
    IngestionVectorStore,
    LLM,
    Reranker,
    VectorStore,
)
from raglight.models import Document


class CustomLLM:
    def generate(self, input: dict) -> str:
        return str(input["question"])

    def generate_streaming(self, input: dict, callbacks=None):
        yield self.generate(input)


class CustomEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class CustomVectorStore:
    def similarity_search(self, question: str, k: int = 5) -> list[Document]:
        return [Document(page_content=question)]


class CustomIngestionVectorStore(CustomVectorStore):
    def add_documents(self, documents: list[Document]) -> None:
        self.documents = list(documents)


class CustomReranker:
    def rerank(
        self, question: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        return list(reversed(documents))[:top_k]


class CustomDocumentProcessor:
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> dict[str, list[Document]]:
        return {"chunks": [Document(page_content=file_path)], "classes": []}


class ExtensionContractTests(unittest.TestCase):
    def test_contract_namespace_exports_only_supported_extension_points(self):
        self.assertEqual(
            [
                "DocumentProcessor",
                "EmbeddingModel",
                "IngestionVectorStore",
                "LLM",
                "Reranker",
                "VectorStore",
            ],
            contracts.__all__,
        )

    def test_custom_components_conform_without_inheriting_from_raglight(self):
        components = (
            (CustomLLM(), LLM),
            (CustomEmbeddingModel(), EmbeddingModel),
            (CustomVectorStore(), VectorStore),
            (CustomIngestionVectorStore(), IngestionVectorStore),
            (CustomReranker(), Reranker),
            (CustomDocumentProcessor(), DocumentProcessor),
        )

        for component, contract in components:
            with self.subTest(contract=contract.__name__):
                self.assertIsInstance(component, contract)
                self.assertNotIn(contract, type(component).__mro__)

    def test_contracts_import_without_optional_dependencies(self):
        script = textwrap.dedent("""
            import sys
            import raglight.contracts

            forbidden = (
                "langchain",
                "langgraph",
                "chromadb",
                "qdrant_client",
                "sentence_transformers",
                "openai",
                "streamlit",
                "fastapi",
            )
            loaded = [name for name in forbidden if name in sys.modules]
            raise SystemExit(f"unexpected imports: {loaded}" if loaded else 0)
            """)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)


class BuiltInContractTests(unittest.TestCase):
    def test_all_builtin_llms_implement_the_llm_contract(self):
        from raglight.integrations.bedrock import BedrockLLM
        from raglight.integrations.gemini import GeminiLLM
        from raglight.integrations.lmstudio import LMStudioLLM
        from raglight.integrations.mistral import MistralLLM
        from raglight.integrations.ollama import OllamaLLM
        from raglight.integrations.openai import OpenAILLM

        implementations = (
            BedrockLLM,
            GeminiLLM,
            LMStudioLLM,
            MistralLLM,
            OllamaLLM,
            OpenAILLM,
        )
        self._assert_implementations(
            implementations,
            LLM,
            (("generate", ({"question": "test"},)), ("generate_streaming", ({},))),
        )

    def test_all_builtin_embedding_models_implement_the_embedding_contract(self):
        from raglight.integrations.bedrock import BedrockEmbeddings
        from raglight.integrations.gemini import GeminiEmbeddings
        from raglight.integrations.huggingface import HuggingFaceEmbeddings
        from raglight.integrations.ollama import OllamaEmbeddings
        from raglight.integrations.openai import OpenAIEmbeddings

        implementations = (
            BedrockEmbeddings,
            GeminiEmbeddings,
            HuggingFaceEmbeddings,
            OllamaEmbeddings,
            OpenAIEmbeddings,
        )
        self._assert_implementations(
            implementations,
            EmbeddingModel,
            (("embed_documents", (["test"],)), ("embed_query", ("test",))),
        )

    def test_all_builtin_vector_stores_implement_the_vector_store_contract(self):
        from raglight.integrations.chroma import ChromaVectorStore
        from raglight.integrations.qdrant import QdrantVectorStore

        self._assert_implementations(
            (ChromaVectorStore, QdrantVectorStore),
            VectorStore,
            (("similarity_search", ("question", 3)),),
        )

        self._assert_implementations(
            (ChromaVectorStore, QdrantVectorStore),
            IngestionVectorStore,
            (
                ("similarity_search", ("question", 3)),
                ("add_documents", ([],)),
            ),
        )

    def test_all_builtin_document_processors_implement_the_processor_contract(self):
        from raglight.integrations.documents import (
            CodeProcessor,
            PDFProcessor,
            TextProcessor,
            VlmPDFProcessor,
        )

        self._assert_implementations(
            (CodeProcessor, PDFProcessor, TextProcessor, VlmPDFProcessor),
            DocumentProcessor,
            (("process", ("document.txt", 500, 50)),),
        )

    def test_builtin_cross_encoder_implements_the_reranker_contract(self):
        from raglight.integrations.huggingface import HuggingFaceReranker

        self._assert_implementations(
            (HuggingFaceReranker,),
            Reranker,
            (("rerank", ("question", [], 3)),),
        )

    def _assert_implementations(self, implementations, contract, method_calls):
        for implementation in implementations:
            with self.subTest(implementation=implementation.__name__):
                instance = object.__new__(implementation)
                self.assertIsInstance(instance, contract)
                for method_name, arguments in method_calls:
                    inspect.signature(getattr(implementation, method_name)).bind(
                        instance, *arguments
                    )


if __name__ == "__main__":
    unittest.main()
