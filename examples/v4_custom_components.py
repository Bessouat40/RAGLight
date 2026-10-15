"""Run the v4 public pipeline with structural custom components."""

from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document


class CustomEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class CustomVectorStore:
    def similarity_search(self, question: str, k: int = 5) -> list[Document]:
        documents = [
            Document(
                page_content="RAGLight supports injected components.",
                metadata={"source": "custom-store"},
            ),
            Document(page_content="No Builder registration is required."),
        ]
        return documents[:k]


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
    pipeline = RAGPipeline(
        llm=CustomLLM(),
        embeddings=CustomEmbeddings(),
        vector_store=CustomVectorStore(),
        reranker=CustomReranker(),
        config=PipelineConfig(top_k=2),
    )

    print(pipeline.generate("Can I use custom components?"))


if __name__ == "__main__":
    main()
