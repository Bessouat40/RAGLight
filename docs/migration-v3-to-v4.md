# Migrating from RAGLight v3 to v4

RAGLight v4 intentionally replaces the overlapping v3 construction APIs with
one pipeline API. Implemented replacements are executable today; later
capabilities are labeled as planned.

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
from raglight import PipelineConfig, RAGPipeline
from raglight.integrations.openai import OpenAIEmbeddings, OpenAILLM
from raglight.integrations.qdrant import QdrantVectorStore

embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small")
pipeline = RAGPipeline(
    llm=OpenAILLM(model_name="gpt-4.1-mini"),
    embeddings=embeddings,
    vector_store=QdrantVectorStore(
        collection_name="docs",
        embeddings_model=embeddings,
    ),
    config=PipelineConfig(top_k=5),
)
```

The v4 pipeline receives ready component instances. Selecting and configuring
provider SDKs is not pipeline behavior.

### Replace implementation inheritance with structural contracts

Custom v3 components commonly inherited from an abstract base class. In v4,
implement the matching protocol methods directly:

```python
from raglight.contracts import LLM


class MyLLM:
    def generate(self, input: dict) -> str:
        return "custom answer"

    def generate_streaming(self, input: dict):
        yield self.generate(input)


custom_llm: LLM = MyLLM()
```

The annotation is optional and no registration or inheritance is required.
The legacy ABCs were removed. See the [complete custom-component
example](../examples/v4_custom_components.py) for
LLM, embeddings, vector-store, reranker, and document-processor contracts.

### Replace the two-config constructor

v3:

```python
pipeline = RAGPipeline(rag_config, vector_store_config)
```

v4 applications either inject components, as above, or use the explicit
serializable registry path:

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import default_registries

pipeline = RAGPipeline.from_config(
    PipelineConfig.from_file("raglight.toml"),
    registries=default_registries,
)
```

Unlike v3 settings and builder branches, each configured component has a
registry name and provider-owned options. Custom factories may register a name,
but custom component instances should normally be injected directly.

`RAGConfig`, `VectorStoreConfig`, and `AgenticRAGConfig` are not carried into
the standard v4 API as separate overlapping configuration objects.

## Agentic pipelines (planned)

Standard RAG and agentic execution have separate v4 entry points. Do not turn a
standard pipeline into an agent with a mode flag.

v3:

```python
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline

agent = AgenticRAGPipeline(agentic_config, vector_store_config)
```

Planned v4 API (not implemented yet):

```python
from raglight.agentic import AgenticPipeline

agent = AgenticPipeline(
    llm=my_llm,
    embeddings=my_embeddings,
    vector_store=my_vector_store,
    tools=my_tools,
    max_steps=4,
)
```

`AgenticPipeline` may reuse the same component implementations, response
models, events, and session semantics as `RAGPipeline`. Agent-only concepts
such as tools, MCP servers, execution limits, and termination policy remain in
`raglight.agentic` and its optional dependency group.

## Imports

| v3 import or concept | v4 replacement |
| --- | --- |
| `raglight.rag.simple_rag_api.RAGPipeline` | `raglight.RAGPipeline` |
| `raglight.rag.rag.RAG` | `raglight.RAGPipeline` |
| `raglight.rag.builder.Builder` | direct component injection |
| `raglight.rag.simple_agentic_rag_api.AgenticRAGPipeline` | `raglight.agentic.AgenticPipeline` |
| `raglight.config.agentic_rag_config.AgenticRAGConfig` | agentic constructor arguments or configuration under `raglight.agentic` |
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
events = response.events
timings = response.timings
trace_id = response.trace_id
```

Use `response.to_dict()` when a transport-friendly representation is needed.
Execution events report completed retrieval, optional reranking, and generation
stages without exposing implementation logs.

## Conversation history

v3 stores history on the pipeline. A shared pipeline can therefore mix state
between users.

v4 pipelines are stateless and accept explicit history:

```python
response = pipeline.generate(
    question,
    history=[{"role": "user", "content": "..."}],
)
```

Use a session when RAGLight should retain completed turns for one conversation:

```python
session = pipeline.create_session(session_id="user-123")
response = session.generate("Remember this conversation")
chunks = session.stream("Continue it")
```

Each session owns its history and can share the configured pipeline with other
sessions. Generation failures and incomplete streams do not commit partial
turns. Use `session.history` for a detached snapshot and `session.clear()` to
reset the messages without changing the session identifier.

## Directory ingestion

v3 commonly exposes the underlying vector store:

```python
pipeline.vector_store.ingest(data_path="./docs")
```

v4 owns ingestion at the application boundary:

```python
from raglight import RAGPipeline
from raglight.integrations import default_registries
from raglight.models import FolderSource

pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    document_processors=default_registries.document_processors,
)

result = pipeline.ingest([FolderSource(path="./docs")])
```

This keeps document processing and persistence orchestration out of the public
vector-store implementation. The result explicitly reports processed,
indexed, skipped, and failed files; adapter-level `ingest(...)` and
`custom_processors` constructor arguments were removed. Register a custom
processor under its file extension in the injected processor registry.

## Compatibility

v4 does not provide compatibility shims for v3 deep imports, builders,
settings, delivery applications, or agentic orchestration. Migrate to public
v4 imports before upgrading.
