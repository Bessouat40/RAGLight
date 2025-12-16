# Getting Started

This guide walks you through creating your **first RAG pipeline** using RAGLight.

The goal is to show a **minimal, explicit setup** with no hidden behavior.

---

## 1. Define your knowledge base

A knowledge base defines the data that will be indexed into the vector store.

RAGLight supports multiple data sources, such as local folders and GitHub repositories.

```python
from raglight.models.data_source_model import FolderSource, GitHubSource

knowledge_base = [
    FolderSource(path="./data"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
]
```

---

## 2. Configure the vector store

The vector store is responsible for embedding and storing documents.

```python
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings

vector_store_config = VectorStoreConfig(
    embedding_model=Settings.DEFAULT_EMBEDDINGS_MODEL,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory="./vector_db",
    collection_name=Settings.DEFAULT_COLLECTION_NAME
)
```

---

## 3. Configure the RAG pipeline

Define how documents are retrieved and how the LLM generates answers.

```python
from raglight.config.rag_config import RAGConfig

rag_config = RAGConfig(
    llm=Settings.DEFAULT_LLM,
    provider=Settings.OLLAMA,
    k=5
)
```

---

## 4. Build the pipeline

Building the pipeline indexes your knowledge base into the vector store.

```python
from raglight.rag.simple_rag_api import RAGPipeline

pipeline = RAGPipeline(
    rag_config,
    vector_store_config,
    knowledge_base=knowledge_base
)

pipeline.build()
```

This step may take some time depending on the size of your documents.

---

## 5. Query the pipeline

Once built, you can generate answers using natural language queries.

```python
response = pipeline.generate(
    "How can I create a simple RAG pipeline using RAGLight?"
)

print(response)
```

---

## What happens internally?

1. Documents are loaded and chunked
2. Embeddings are generated
3. Vectors are stored in the vector database
4. Relevant chunks are retrieved for each query
5. The LLM generates an answer using retrieved context

Everything is explicit and configurable.

---

## Next steps

- Explore **Agentic RAG** for multi-step reasoning
- Try **RAT pipelines** for reflection-based answers
- Integrate **MCP servers** for tool usage
- Check **Examples** for advanced usage patterns

➡️ Continue with **Documentation** to explore configurations and APIs in detail.
