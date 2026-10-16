# RAGLight

RAGLight 4 is a small, typed Python toolkit for composing retrieval-augmented
generation pipelines. The core package has no third-party runtime dependency:
applications inject an LLM, an embedding model, a vector store, and optionally
a reranker through public protocols.

> RAGLight 4 is a breaking release. The v3 builder, global settings, agentic
> runtime, API server, CLI, and UI have been removed instead of being carried as
> compatibility layers.

## Requirements

- Python 3.11 or newer
- One optional integration extra when using a built-in provider

```bash
pip install raglight
pip install "raglight[ollama,chroma,documents]"
```

Available extras are `openai`, `ollama`, `mistral`, `gemini`, `bedrock`,
`huggingface`, `documents`, `chroma`, and `qdrant`.

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
from raglight.models import Document, ExecutionEvent, FolderSource, RAGResponse
```

The contracts are structural protocols. Custom components do not inherit from
RAGLight classes and do not need registry entries.

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document


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
        self, question: str, k: int = 5, filter: dict[str, object] | None = None
    ) -> list[Document]:
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

A session commits a turn only after generation or streaming completes. Failed
or abandoned streams do not leave partial conversation state behind. Separate
sessions share configured components without sharing history.

See [`examples/v4_custom_components.py`](examples/v4_custom_components.py) for
a complete executable example.

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
[`examples/v4_directory_ingestion.py`](examples/v4_directory_ingestion.py) for
a local Ollama and Chroma example.

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

### Registry-backed configuration

Direct component injection remains the primary API. Registries provide an
optional path for serializable JSON or TOML configuration without
provider-specific branches in the pipeline:

```toml
top_k = 3

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
[`examples/v4_registry_configuration.py`](examples/v4_registry_configuration.py)
for an offline custom-registry example.

## Architecture

Dependencies point inward:

```text
integrations -> registry/contracts/models <- pipeline
```

- registry primitives, contracts, and models are dependency-free.
- `pipeline` depends only on public contracts and models.
- built-in integrations own their optional SDK dependencies.
- agentic orchestration is intentionally separate from the classic RAG API and
  will return as a dedicated v4 capability.

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
