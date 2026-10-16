"""Run the v4 Agentic RAG API with local Ollama and Chroma components."""

from tempfile import TemporaryDirectory

from raglight.agentic import AgenticPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.models import Document

embeddings = OllamaEmbeddings(model_name="nomic-embed-text:latest")
llm = OllamaLLM(model_name="qwen3:0.6b", preload_model=False)

with TemporaryDirectory() as database:
    vector_store = ChromaVectorStore(
        collection_name="agentic-example",
        embeddings_model=embeddings,
        persist_directory=database,
    )
    vector_store.add_documents(
        [
            Document(
                page_content=(
                    "RAGLight v4 keeps classic RAG and Agentic RAG in separate "
                    "public APIs."
                ),
                metadata={"source": "inline-example"},
            )
        ]
    )

    pipeline = AgenticPipeline(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        max_steps=4,
    )
    response = pipeline.generate(
        "How does RAGLight v4 separate classic and agentic execution?"
    )

print(response.answer)
print("Sources:", [source.metadata for source in response.sources])
print("Events:", [event.to_dict() for event in response.events])
