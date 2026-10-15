"""Build a v4 pipeline from named custom factories and serialized values."""

from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import IntegrationRegistries, IntegrationRegistry
from raglight.models import Document


class ExampleLLM:
    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix

    def generate(self, input):
        return self.prefix + input["question"]

    def generate_streaming(self, input):
        yield self.generate(input)


class ExampleEmbeddings:
    def __init__(self, *, dimensions: int = 1) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts):
        return [[float(len(text))] * self.dimensions for text in texts]

    def embed_query(self, text):
        return [float(len(text))] * self.dimensions


class ExampleVectorStore:
    def __init__(self, *, embeddings_model, content: str) -> None:
        self.embeddings_model = embeddings_model
        self.content = content

    def similarity_search(self, question, k=5):
        return [Document(page_content=self.content)][:k]


registries = IntegrationRegistries(
    llms=IntegrationRegistry("LLM"),
    embeddings=IntegrationRegistry("embedding"),
    vector_stores=IntegrationRegistry("vector store"),
    rerankers=IntegrationRegistry("reranker"),
    document_processors=IntegrationRegistry("document processor"),
)
registries.llms.register("example", ExampleLLM)
registries.embeddings.register("example", ExampleEmbeddings)
registries.vector_stores.register("example", ExampleVectorStore)

config = PipelineConfig.from_dict(
    {
        "top_k": 1,
        "llm": {"name": "example", "options": {"prefix": "Generated: "}},
        "embeddings": {"name": "example", "options": {"dimensions": 2}},
        "vector_store": {
            "name": "example",
            "options": {"content": "Registry-backed context."},
        },
    }
)
pipeline = RAGPipeline.from_config(config, registries=registries)

print(pipeline.generate("How does named construction work?"))
