# Migrating from RAGLight v3 to v4

RAGLight v4 intentionally replaces the overlapping v3 construction APIs with
one pipeline API. This guide records the target migration before implementation
so each v4 change has a stable destination.

The examples on this page describe the target v4 API. They will become
executable as the linked v4 issues land.

## Construction

### Replace `Builder` with component injection

v3:

```python
from raglight.rag.builder import Builder

pipeline = (
    Builder()
    .with_embeddings("openai", model_name="text-embedding-3-small")
    .with_vector_store("qdrant", collection_name="docs")
    .with_llm("openai", model_name="gpt-4.1-mini")
    .build_rag(k=5)
)
```

v4:

```python
from raglight import RAGPipeline
from raglight.integrations.openai import OpenAIEmbeddings, OpenAILLM
from raglight.integrations.qdrant import QdrantVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pipeline = RAGPipeline(
    llm=OpenAILLM(model="gpt-4.1-mini"),
    embeddings=embeddings,
    vector_store=QdrantVectorStore(
        collection="docs",
        embeddings=embeddings,
    ),
    top_k=5,
)
```

The v4 pipeline receives ready component instances. Selecting and configuring
provider SDKs is not pipeline behavior.

### Replace the two-config constructor

v3:

```python
pipeline = RAGPipeline(rag_config, vector_store_config)
```

v4 applications either inject components, as above, or use the explicit
serializable factory path:

```python
from raglight import PipelineConfig, RAGPipeline

pipeline = RAGPipeline.from_config(PipelineConfig.from_file("raglight.toml"))
```

`RAGConfig`, `VectorStoreConfig`, and `AgenticRAGConfig` are not carried into
the standard v4 API as separate overlapping configuration objects.

## Imports

| v3 import or concept | v4 replacement |
| --- | --- |
| `raglight.rag.simple_rag_api.RAGPipeline` | `raglight.RAGPipeline` |
| `raglight.rag.rag.RAG` | `raglight.RAGPipeline` |
| `raglight.rag.builder.Builder` | direct component injection |
| `raglight.config.rag_config.RAGConfig` | `raglight.PipelineConfig` when configuration files are required |
| `raglight.config.vector_store_config.VectorStoreConfig` | integration constructor arguments or `PipelineConfig` |
| `raglight.llm.llm.LLM` | `raglight.contracts.LLM` |
| `raglight.embeddings.embeddings_model.EmbeddingsModel` | `raglight.contracts.EmbeddingModel` |
| `raglight.vectorstore.vector_store.VectorStore` | `raglight.contracts.VectorStore` |
| root concrete-provider imports | `raglight.integrations.<provider>` |
| `Settings.*` provider constants | integration names in configuration, or direct integration constructors |

## Generation results

v3 returns generated text directly:

```python
answer = pipeline.generate(question)
```

v4 returns a structured response:

```python
response = pipeline.generate(question)
answer = response.answer
sources = response.sources
```

See #159 for the response and execution-event model.

## Conversation history

v3 stores history on the pipeline. A shared pipeline can therefore mix state
between users.

v4 pipelines are stateless. Pass history explicitly or use an isolated
session:

```python
session = pipeline.create_session()
response = session.generate("Remember this conversation")
```

See #160 for concurrency, session lifetime, and streaming behavior.

## Ingestion

v3 commonly exposes the underlying vector store:

```python
pipeline.vector_store.ingest(data_path="./docs")
```

v4 owns ingestion at the application boundary:

```python
from raglight.models import FolderSource

result = pipeline.ingest([FolderSource(path="./docs")])
```

This keeps document processing and persistence orchestration out of the public
vector-store implementation.

## Compatibility during development

Before the v4 release, internal compatibility shims may keep v3 tests and
examples working while packages move. They are migration scaffolding, not v4
API commitments. The final v4 release removes undocumented deep imports and
does not load v3 providers from the root package.

Each implementation pull request must update this guide when its actual
behavior differs from the target examples.
