"""Ingest one local directory and query it with Ollama and Chroma."""

from pathlib import Path
from tempfile import TemporaryDirectory

from raglight import RAGPipeline
from raglight.integrations import default_registries
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.models import FolderSource

SOURCE_DIRECTORY = Path("/absolute/path/to/your/knowledge-directory")
QUESTION = "What are the main topics in these documents?"


def main() -> None:
    embeddings = OllamaEmbeddings(model_name="nomic-embed-text:latest")
    llm = OllamaLLM(model_name="qwen3:0.6b", preload_model=False)

    with TemporaryDirectory() as database:
        vector_store = ChromaVectorStore(
            collection_name="local-directory-test",
            embeddings_model=embeddings,
            persist_directory=database,
        )
        pipeline = RAGPipeline(
            llm=llm,
            embeddings=embeddings,
            vector_store=vector_store,
            document_processors=default_registries.document_processors,
        )

        ingestion = pipeline.ingest([FolderSource(path=SOURCE_DIRECTORY)])
        response = pipeline.generate(QUESTION)

        print(ingestion.to_dict())
        print(response.answer)
        print("Sources:", [source.metadata for source in response.sources])


if __name__ == "__main__":
    main()
