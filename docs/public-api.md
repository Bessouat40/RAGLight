# Public API

This document is the normative design for the RAGLight v4 Python API. It
describes the target API, not the API currently implemented by v3. The
implementation will land incrementally; until the v4 release, examples must
label target-v4 code as such.

## Implementation status

The first v4 slice currently implements:

- `raglight.RAGPipeline` with keyword-only component injection;
- immutable `raglight.PipelineConfig(top_k=...)`;
- stateless `generate(..., history=...)` returning text;
- stateless `stream(..., history=...)` yielding text chunks;
- a root import that loads no concrete provider or delivery package.

The rest of this document defines the destination for subsequent milestone
issues. Contracts and reranking land in #157, registry-backed configuration in
#158, structured responses and events in #159, sessions in #160, and ingestion
orchestration as the architecture migration progresses.

## Design principles

- There is one pipeline type for standard retrieval-augmented generation.
- Passing component instances is the primary construction mechanism.
- Name-based provider configuration is a convenience factory, not the
  extension mechanism.
- Conversation state belongs to a session or caller, never to a shared
  pipeline.
- Responses, documents, messages, and execution events are independent of
  HTTP, CLI, LangChain, and provider SDKs.
- Importing the root package never imports an optional provider.
- Public names are exported deliberately and tested through their public path.

## Public namespaces

### Common API: `raglight`

Only the two names required to construct a normal pipeline are exported from
the package root:

```python
from raglight import PipelineConfig, RAGPipeline
```

`RAGPipeline` is the single standard pipeline abstraction. `PipelineConfig`
contains provider-independent pipeline behavior. Registry-backed component
specifications will extend the named construction path in #158.

No concrete provider, builder, low-level graph, global settings object, or
delivery-layer type is exported from `raglight`.

### Domain models: `raglight.models`

Transport-independent values are available from one documented namespace:

```python
from raglight.models import (
    Document,
    FolderSource,
    GitHubSource,
    IngestionResult,
    Message,
    RAGEvent,
    RAGResponse,
)
```

The exact fields of response and event models are finalized by #159. Session
behavior is finalized by #160. These issues may add fields, but they must not
move the models to transport-specific namespaces.

### Extension contracts: `raglight.contracts`

Third-party components depend only on stable structural contracts:

```python
from raglight.contracts import (
    DocumentProcessor,
    EmbeddingModel,
    LLM,
    Reranker,
    VectorStore,
)
```

Implementations do not need to inherit from a RAGLight base class. Contract
method signatures and shared contract tests are delivered by #157. Adding a
provider must not require editing `RAGPipeline`.

### Optional integrations: `raglight.integrations`

Concrete adapters live below an explicit provider namespace and are never
re-exported from the package root:

```python
from raglight.integrations.openai import OpenAIEmbeddings, OpenAILLM
from raglight.integrations.qdrant import QdrantVectorStore
```

Importing an integration may require its matching extra. Importing `raglight`,
`raglight.models`, or `raglight.contracts` must not. Registry APIs introduced
by #158 also live under `raglight.integrations`.

### Optional capabilities

Features that are not part of the basic RAG workflow use dedicated namespaces:

- `raglight.agentic` for `AgenticPipeline`, its configuration, tools, and MCP
  support;
- `raglight.observability` for tracing integrations;
- `raglight.server`, `raglight.cli`, and `raglight.ui` for delivery layers.

These namespaces are not imported by the root package and may require extras.

## Standard and agentic pipelines

RAGLight exposes two intentionally separate application APIs:

```python
from raglight import RAGPipeline
from raglight.agentic import AgenticPipeline
```

`RAGPipeline` owns a deterministic retrieval workflow:

```text
retrieve -> optional rerank -> generate
```

`AgenticPipeline` owns an iterative decision loop that may select retrieval or
other tools until it reaches an answer or a configured execution limit. Its
tool registry, step budget, MCP support, and termination behavior do not belong
on `RAGPipeline`.

The two APIs share domain models and component contracts where their semantics
are identical:

- both may receive `LLM`, `EmbeddingModel`, and `VectorStore` implementations;
- both return `RAGResponse` and emit `RAGEvent` values;
- both use caller-owned history or isolated sessions;
- both use integrations from `raglight.integrations`.

They do not inherit from one another, and `RAGPipeline` does not accept an
`agentic=True` mode flag. Shared implementation belongs in internal application
services, not in a public base class. Installing or importing standard RAG must
not require the `agentic` extra, MCP, or agent-framework dependencies.

## Pipeline construction

Dependency injection is the canonical API:

```python
from raglight import RAGPipeline

pipeline = RAGPipeline(
    llm=my_llm,
    embeddings=my_embeddings,
    vector_store=my_vector_store,
    config=PipelineConfig(top_k=5),
)
```

Provider-independent behavior such as retrieval count is supplied through
`PipelineConfig`; provider credentials and SDK clients remain owned by
integration instances. Optional reranker injection is added with the extension
contracts in #157.

After registries land in #158, serializable configuration will use a named
constructor so it cannot be confused with dependency injection:

```python
from raglight import PipelineConfig, RAGPipeline

pipeline = RAGPipeline.from_config(PipelineConfig.from_file("raglight.toml"))
```

`from_config` resolves registered integrations. Custom objects never need a
registry entry.

## Generation and sessions

Stateless generation is the default:

```python
response = pipeline.generate("What is RAGLight?")
print(response.answer)
```

Callers may pass explicit history, or create an isolated conversation:

```python
session = pipeline.create_session()
first = session.generate("What is RAGLight?")
second = session.generate("How do I extend it?")
```

Streaming uses the same response and event model instead of returning
provider-specific chunks:

```python
for event in pipeline.stream("What is RAGLight?"):
    handle(event)
```

The final session and streaming signatures are implemented by #159 and #160.

## Ingestion

Ingestion is a pipeline use case, not a vector-store side channel:

```python
from raglight.models import FolderSource

result = pipeline.ingest([FolderSource(path="./docs")])
```

Application code must not need `pipeline.vector_store.ingest(...)` or direct
access to a builder-created internal object.

## Stability rules

- Names documented in this file are public once implemented and released.
- Undocumented deep imports are internal even when Python can import them.
- Public parameters are keyword-only unless positional use is explicitly
  documented.
- Public inputs and return values have complete type annotations.
- A public name cannot import an optional dependency as a side effect.
- Removal or incompatible behavior requires a major release.
- Deprecations identify a replacement and an intended removal version.
- Every public import and workflow has an offline automated test.

Adding a root export is a compatibility commitment. It requires an API review,
documentation, migration impact assessment, and a public-import test.

## Explicitly non-public in v4

The following v3 concepts are not part of the v4 public API:

- `Builder` and its provider-specific conditional methods;
- the low-level `RAG` and `AgenticRAG` graph implementations;
- `Settings` as a global bag of provider constants and credentials;
- concrete providers imported directly from `raglight`;
- transport request/response models;
- LangChain or LangGraph objects used internally;
- mutable pipeline-owned conversation history.

The v3 `AgenticRAGPipeline` is replaced by
`raglight.agentic.AgenticPipeline`; agentic behavior remains public but is no
longer mixed into the standard RAG namespace or configuration model.

See [Migrating from v3 to v4](migration-v3-to-v4.md) for replacements.

## Review checklist

- Can the behavior be expressed through an existing public abstraction?
- Is this a domain concept or an integration/delivery implementation detail?
- Does importing the symbol load an optional dependency?
- Can the API evolve without adding provider-specific branches?
- Is the behavior covered using only public imports?
- Is its migration impact documented?
