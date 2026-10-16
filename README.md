# RAGLight

RAGLight 4 is a small, typed Python toolkit for composing classic and agentic
retrieval-augmented generation pipelines. The core package has no third-party
runtime dependency: applications inject an LLM, an embedding model, a vector
store, and optional behavior through public structural protocols.

> RAGLight 4 is a breaking release. The v3 builder, global settings, API server,
> CLI, UI, and legacy agentic runtime were removed instead of being carried as
> compatibility layers. The v4 agentic API is a new implementation.

## Requirements

- Python 3.11 or newer
- One optional integration extra when using a built-in provider

```bash
pip install raglight
pip install "raglight[ollama,chroma,documents]"
```

Available extras are `openai`, `ollama`, `mistral`, `gemini`, `bedrock`,
`huggingface`, `documents`, `chroma`, `qdrant`, `langfuse`, and `agentic-mcp`.

## Copy-paste examples

The [examples index](examples/README.md) contains complete runnable workflows
for classic RAG with Ollama, OpenAI, and OpenAI-compatible endpoints; recursive
directory ingestion; Agentic RAG; MCP tools; custom agents; and custom
components. Each provider example includes its exact installation and execution
commands, while offline examples require no credentials or external services.

## Public API

The stable core is deliberately narrow:

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.contracts import (
    DocumentProcessor,
    EmbeddingModel,
    IngestionVectorStore,
    LLM,
    Reranker,
    VectorStore,
)
from raglight.models import (
    And,
    Document,
    Eq,
    ExecutionEvent,
    FolderSource,
    In,
    MatchNone,
    Or,
    RAGResponse,
    RetrievalFilter,
)
```

The contracts are structural protocols. Custom components do not inherit from
RAGLight classes and do not need registry entries.

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document, RetrievalFilter


class MyLLM:
    def generate(self, input: dict[str, object]) -> str:
        return "RAGLight is extended through structural component contracts."


class MyEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class MyVectorStore:
    def similarity_search(
        self,
        question: str,
        k: int = 5,
        *,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[Document]:
        if retrieval_filter is not None:
            raise NotImplementedError("this store does not support filtering")
        return [Document(page_content="RAGLight uses dependency injection.")][:k]

    def add_documents(self, documents: list[Document]) -> None:
        pass


pipeline = RAGPipeline(
    llm=MyLLM(),
    embeddings=MyEmbeddings(),
    vector_store=MyVectorStore(),
    config=PipelineConfig(top_k=3),
)

response = pipeline.generate("How is RAGLight extended?")
print(response.answer)
print(response.sources)
print(response.events)
```

`generate()` returns an immutable `RAGResponse`. Its `answer`, `sources`,
structured execution `events`, per-stage `timings`, `metadata`, and `trace_id`
can be consumed directly or converted into transport-friendly values with
`response.to_dict()`. Streaming continues to yield text chunks through
`pipeline.stream(...)`.

Use provider-neutral filters when retrieval must stay inside an application
scope. Qdrant and Chroma translate the same filter tree at their integration
boundaries:

```python
from raglight.models import And, Eq, In

scope = And(
    Eq("workspace_id", "ws-123"),
    In("file_id", ["file-a", "file-b"]),
)
response = pipeline.generate("Summarize these files", retrieval_filter=scope)
```

`None` keeps trusted standalone retrieval unrestricted. `MatchNone()` always
returns no sources without querying a backend. Applications must calculate
authorization scopes server-side; RAGLight does not implement users or ACLs.

The pipeline itself is stateless and safe to share between users. Pass
`history` explicitly for stateless requests, or create an isolated session when
RAGLight should retain completed turns:

```python
response = pipeline.generate("One-off question", history=[])

session = pipeline.create_session(session_id="user-123")
first = session.generate("Remember this question")
chunks = list(session.stream("What did I ask?"))

print(session.history)
session.clear()
```

Conversational query reformulation is opt-in and uses the configured LLM only
when history is present:

```python
pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    config=PipelineConfig(top_k=5, reformulate_queries=True),
)
```

The rewritten value is used for retrieval and reranking; generation still sees
the original question and history. Successful responses expose the rewritten
query in `response.metadata["retrieval_query"]` and add a `reformulation` event.

A session commits a turn only after generation or streaming completes. Failed
or abandoned streams do not leave partial conversation state behind. Separate
sessions share configured components without sharing history.

See
[`examples/extending/custom_components.py`](examples/extending/custom_components.py)
for a complete executable example.

## Agentic RAG

Agentic execution has a deliberately separate public namespace and does not
change the deterministic `RAGPipeline`:

```python
from raglight.agentic import AgenticPipeline, FunctionTool

pipeline = AgenticPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    tools=[
        FunctionTool(
            "weather",
            "Return weather for a city.",
            lambda arguments: weather_client.lookup(arguments["city"]),
            {"type": "object", "required": ["city"]},
        )
    ],
    max_steps=4,
)

response = pipeline.generate("Use the knowledge base and weather tool to answer")
```

The built-in `retrieve` tool uses the injected vector store and optional
reranker. The default planner asks the injected LLM for strict JSON decisions.
Pass `agent=my_agent` to replace it with any object implementing the structural
`Agent.decide(context)` contract; no inheritance or framework adapter is
required. Tool failures become observations that the agent can recover from,
while an unfinished loop raises `AgentStepLimitError` at the configured budget.

Use `pipeline.create_session(...)` for isolated agentic history. Async tools are
supported through `await pipeline.agenerate(...)`. MCP is an optional adapter:

```bash
pip install "raglight[agentic-mcp]"
```

`raglight.agentic.mcp.MCPTool` wraps a tool on a caller-managed, connected MCP
client. See the [Agentic RAG examples](examples/README.md#agentic-rag) and the
[public API guide](docs/public-api.md) for complete examples.

### Directory ingestion

Directory ingestion belongs to the public pipeline, not to a vector-store
adapter. Inject the lazy document-processor registry, then pass one or more
folder sources:

```python
from raglight.integrations import default_registries
from raglight.models import FolderSource

pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    document_processors=default_registries.document_processors,
)
result = pipeline.ingest([FolderSource(path="./knowledge")])

print(result.processed_count, result.indexed_count)
print(result.skipped, result.failures)
```

Traversal is recursive and deterministic by default. Common generated
directories and symlinks are skipped, unsupported extensions are reported,
and one failed file does not discard successful files. Use
`ignored_directories`, `recursive`, `chunk_size`, and `chunk_overlap` on each
`FolderSource` to customize discovery and processing. See
[`examples/classic/directory_ingestion.py`](examples/classic/directory_ingestion.py)
for a local Ollama and Chroma example.

## Built-in integrations

Import integrations explicitly so importing `raglight` never loads an optional
provider SDK:

```python
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations import default_registries

embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
vector_store = ChromaVectorStore(
    collection_name="knowledge",
    embeddings_model=embeddings,
    persist_directory="./raglight-db",
)
llm = OllamaLLM(model_name="llama3.1:8b")

pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    document_processors=default_registries.document_processors,
)
```

Provider modules and public classes:

| Module | Classes |
| --- | --- |
| `raglight.integrations.openai` | `OpenAILLM`, `OpenAIEmbeddings` |
| `raglight.integrations.ollama` | `OllamaLLM`, `OllamaEmbeddings` |
| `raglight.integrations.gemini` | `GeminiLLM`, `GeminiEmbeddings` |
| `raglight.integrations.bedrock` | `BedrockLLM`, `BedrockEmbeddings` |
| `raglight.integrations.mistral` | `MistralLLM` |
| `raglight.integrations.lmstudio` | `LMStudioLLM` |
| `raglight.integrations.huggingface` | `HuggingFaceEmbeddings`, `HuggingFaceReranker` |
| `raglight.integrations.chroma` | `ChromaVectorStore` |
| `raglight.integrations.qdrant` | `QdrantVectorStore` |
| `raglight.integrations.documents` | `TextProcessor`, `CodeProcessor`, `PDFProcessor`, `VlmPDFProcessor` |
| `raglight.integrations.langfuse` | `LangfuseTracer` |

### Native Langfuse tracing

```bash
pip install "raglight[langfuse]"
```

```python
from raglight.integrations.langfuse import LangfuseTracer

tracer = LangfuseTracer()
pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    tracer=tracer,
)
```

The SDK reads its standard `LANGFUSE_*` environment variables. Classic RAG
creates nested reformulation, retrieval, reranking, and generation observations.
Agentic RAG creates decision and tool observations through the same `tracer=`
argument. Call `tracer.flush()` before a short-lived process exits.

### Registry-backed configuration

Direct component injection remains the primary API. Registries provide an
optional path for serializable JSON or TOML configuration without
provider-specific branches in the pipeline:

```toml
top_k = 3
reformulate_queries = true

[llm]
name = "ollama"
[llm.options]
model_name = "llama3.1:8b"
preload_model = false

[embeddings]
name = "ollama"
[embeddings.options]
model_name = "nomic-embed-text"

[vector_store]
name = "chroma"
[vector_store.options]
collection_name = "knowledge"
persist_directory = "./raglight-db"
```

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import default_registries

config = PipelineConfig.from_file("raglight.toml")
pipeline = RAGPipeline.from_config(config, registries=default_registries)
```

Custom factories may opt into named configuration:

```python
default_registries.llms.register("my-llm", MyLLM)
```

Registration is never required when passing a component instance directly.
See
[`examples/extending/registry_configuration.py`](examples/extending/registry_configuration.py)
for an offline custom-registry example.

## Architecture

Dependencies point inward:

```text
integrations -> registry/contracts/models <- pipeline
                                      ^---- agentic
```

- registry primitives, contracts, and models are dependency-free.
- `pipeline` and `agentic` are separate sibling application layers.
- neither application layer imports concrete integrations or the other layer.
- built-in integrations own their optional SDK dependencies.
- agentic orchestration is intentionally separate from the classic RAG API.

The detailed boundaries are documented in
[`docs/architecture.md`](docs/architecture.md), and the v3 migration table is in
[`docs/migration-v3-to-v4.md`](docs/migration-v3-to-v4.md).

## Development

The project follows test-driven development. Every behavior change starts with
a failing test, then the smallest implementation, then refactoring while the
suite remains green.

```bash
make install-dev
make check
```

`make check` runs formatting, linting, type checking, architecture rules, unit
tests, integration tests, end-to-end tests, and an isolated wheel validation.
See [`docs/testing.md`](docs/testing.md) and
[`docs/development.md`](docs/development.md) for the contribution workflow.

## License

RAGLight is distributed under the MIT License.
