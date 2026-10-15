"""Run the v4 public pipeline with deterministic custom components."""

from dataclasses import dataclass

from raglight import PipelineConfig, RAGPipeline


@dataclass(frozen=True)
class Document:
    page_content: str


class CustomEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class CustomVectorStore:
    def similarity_search(self, question: str, k: int = 5) -> list[Document]:
        return [Document(page_content="RAGLight supports injected components.")]


class CustomLLM:
    def generate(self, input: dict) -> str:
        return "This answer came from a custom LLM."

    def generate_streaming(self, input: dict, callbacks=None):
        yield "This answer came "
        yield "from a custom LLM."


pipeline = RAGPipeline(
    llm=CustomLLM(),
    embeddings=CustomEmbeddings(),
    vector_store=CustomVectorStore(),
    config=PipelineConfig(top_k=3),
)

print(pipeline.generate("Can I use custom components?"))
