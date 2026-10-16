"""Run the v4 public pipeline with structural custom components."""

from pathlib import Path
from tempfile import TemporaryDirectory

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import IntegrationRegistry
from raglight.models import Document, FolderSource


class CustomEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class CustomVectorStore:
    def __init__(self) -> None:
        self.documents = [
            Document(
                page_content="RAGLight supports injected components.",
                metadata={"source": "custom-store"},
            ),
            Document(page_content="No registry entry is required."),
        ]

    def similarity_search(self, question: str, k: int = 5) -> list[Document]:
        return self.documents[:k]

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)


class CustomReranker:
    def rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        return sorted(documents, key=lambda document: len(document.page_content))[
            :top_k
        ]


class CustomDocumentProcessor:
    def process(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, list[Document]]:
        return {
            "chunks": [Document(page_content=f"Processed {file_path}")],
            "classes": [],
        }


class CustomLLM:
    def generate(self, input: dict) -> str:
        return "This answer came from a custom LLM."

    def generate_streaming(self, input: dict, callbacks=None):
        yield "This answer came "
        yield "from a custom LLM."


def main() -> None:
    processors = IntegrationRegistry("document processor")
    processors.register("custom", CustomDocumentProcessor)
    pipeline = RAGPipeline(
        llm=CustomLLM(),
        embeddings=CustomEmbeddings(),
        vector_store=CustomVectorStore(),
        reranker=CustomReranker(),
        config=PipelineConfig(top_k=2),
        document_processors=processors,
    )

    with TemporaryDirectory() as directory:
        source = Path(directory) / "notes.custom"
        source.write_text("Custom processor input.", encoding="utf-8")
        ingestion = pipeline.ingest([FolderSource(path=directory)])
        print(f"Indexed chunks: {ingestion.indexed_count}")

    response = pipeline.generate("Can I use custom components?")
    print(response.answer)
    print(f"Sources: {len(response.sources)}")
    print(f"Stages: {[event.stage for event in response.events]}")
    print(f"Trace: {response.trace_id}")

    session = pipeline.create_session(session_id="example-conversation")
    session.generate("Remember that I use custom components.")
    print("".join(session.stream("What do I use?")))
    print(f"Committed messages: {len(session.history)}")


if __name__ == "__main__":
    main()
