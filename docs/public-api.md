# Public API

This document is normative for the implemented RAGLight 4 API. Planned APIs
are listed separately and must not be presented as available.

## Root facade

The package root exports exactly two names:

```python
from raglight import PipelineConfig, RAGPipeline
```

`RAGPipeline` coordinates classic retrieval, optional reranking, and
generation. Components are keyword-only constructor arguments. The pipeline is
stateless: callers pass history explicitly to `generate` or `stream`.

```python
pipeline = RAGPipeline(
    llm=my_llm,
    embeddings=my_embeddings,
    vector_store=my_vector_store,
    reranker=my_reranker,  # optional
    config=PipelineConfig(top_k=5),  # optional
)

answer: str = pipeline.generate("What is RAGLight?", history=[])
chunks = pipeline.stream("What is RAGLight?", history=[])
```

`PipelineConfig` is immutable. It owns `top_k`, which must be greater than zero,
and optional serializable component selections used by registry-backed
construction.

## Domain models

```python
from raglight.models import ComponentConfig, Document
```

`Document(page_content: str, metadata: dict[str, object] = {})` is the value
exchanged by retrieval, reranking, and document-processing boundaries.
`ComponentConfig(name, options)` is the provider-independent serialized form
of one named integration.

## Extension contracts

```python
from raglight.contracts import (
    DocumentProcessor,
    EmbeddingModel,
    LLM,
    Reranker,
    VectorStore,
)
```

These are runtime-checkable structural protocols. Implementations do not
inherit from a RAGLight base class and need no registration.

| Contract | Required methods |
| --- | --- |
| `LLM` | `generate(input)`, `generate_streaming(input)` |
| `EmbeddingModel` | `embed_documents(texts)`, `embed_query(text)` |
| `VectorStore` | `similarity_search(question, k=5)` |
| `Reranker` | `rerank(question, documents, top_k)` |
| `DocumentProcessor` | `process(file_path, chunk_size, chunk_overlap)` |

The LLM input contains `question` and `history`. Vector stores and rerankers
exchange `Document` values. A processor returns a dictionary with `chunks` and
`classes` document lists.

## Built-in integrations

Concrete providers are public only from explicit provider modules:

```python
from raglight.integrations.openai import OpenAIEmbeddings, OpenAILLM
from raglight.integrations.qdrant import QdrantVectorStore
```

Provider modules may require their matching optional extra. Importing the root,
contracts, or models must not load those dependencies. Undocumented deep paths
inside `raglight.integrations` are implementation details.

### Integration registries

The registry API is public from the integration namespace:

```python
from raglight.integrations import (
    IntegrationRegistries,
    IntegrationRegistry,
    default_registries,
)
```

Built-in entries are lazy import references. Inspecting registry names does not
load an integration module or SDK. A registry normalizes names, rejects
accidental duplicate registration, and reports available names for failed
lookups.

Custom components can optionally participate in named configuration:

```python
default_registries.llms.register("my-llm", MyLLM)
```

Direct injection does not require registration.

### Serializable construction

`PipelineConfig.from_dict(...)` validates an in-memory mapping.
`PipelineConfig.from_file(...)` supports `.json` and `.toml` files. Named
construction requires explicit registries so the pipeline never imports a
concrete provider:

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.integrations import default_registries

config = PipelineConfig.from_file("raglight.toml")
pipeline = RAGPipeline.from_config(config, registries=default_registries)
```

The required component sections are `llm`, `embeddings`, and `vector_store`;
`reranker` is optional. Each section contains `name` and an optional `options`
mapping. The configured embedding instance is injected into the vector-store
factory as `embeddings_model` and cannot be overridden by serialized data.

## Stability rules

- A documented implemented name is public.
- Root exports are deliberate and tested.
- Public inputs and outputs have type annotations.
- Provider SDK types do not cross core boundaries.
- Adding a provider does not modify `RAGPipeline`.
- Incompatible public behavior requires a major release.
- Every public workflow has offline automated coverage.

The v3 `Builder`, `RAG`, global `Settings`, provider ABCs, agentic runtime,
server, CLI, and UI are removed and are not v4 aliases or compatibility shims.

## Planned capabilities

The following are not implemented in this change:

- structured responses and execution events (#159);
- isolated conversation sessions (#160);
- a separately designed agentic API;
- delivery applications built on the v4 core.

Agentic execution will remain distinct from classic `RAGPipeline`: it will not
use inheritance or an `agentic=True` flag. See
[`architecture.md`](architecture.md) and
[`migration-v3-to-v4.md`](migration-v3-to-v4.md).
