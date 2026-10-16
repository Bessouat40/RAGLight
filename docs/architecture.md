# Architecture

RAGLight 4 has a dependency-free core and explicit optional integrations.

```text
                         +----------------+
                         | contracts/models|
                         | registry core  |
                         +-------^--------+
                                 |
                 +---------------+---------------+---------------+
                 |                               |               |
        +--------+---------+            +--------+---------+  +--+---------+
        |     pipeline     |            |   integrations   |  |  agentic   |
        |  orchestration   |            | provider adapters|  | tool loop  |
        +------------------+            +------------------+  +------------+
```

Dependencies point toward contracts and models. The classic pipeline, agentic
runtime, and integrations are siblings: application layers receive components
through protocols and never import a concrete provider or each other.

## Package ownership

| Package | Responsibility |
| --- | --- |
| `raglight.contracts` | Stable structural protocols for extensions |
| `raglight.models` | Provider- and transport-independent values |
| `raglight.registry` | Provider-independent lazy registry primitives |
| `raglight.pipeline` | Stateless classic orchestration and isolated sessions |
| `raglight.agentic` | Bounded agentic orchestration, tools, and isolated sessions |
| `raglight.integrations` | Optional provider adapters and SDK translation |

`raglight.RAGPipeline` and `raglight.PipelineConfig` are the root facade.
Internal implementation modules are not public merely because Python can
import them.

## Core rules

- contracts, models, and registry primitives do not import integrations or
  optional SDKs.
- `pipeline` imports core registry types, contracts, and models, never
  integrations.
- `agentic` imports shared contracts and models, never `pipeline` or concrete
  integrations.
- importing `raglight` does not load any optional provider package.
- domain data uses `raglight.models.Document`, not an SDK document type.
- configuration for credentials and provider clients stays at the integration
  boundary.
- conversation history is supplied by the caller or owned by one explicit
  session; it is never pipeline or global state.
- a requested retrieval scope reaches every built-in retrieval operation or
  fails closed; it is never silently replaced by an unrestricted search.

These rules are executable in `scripts/check_architecture.py` and
`tests/architecture/`.

## Classic versus agentic RAG

The classic v4 API contains one deterministic flow:

```text
question -> optional reformulation -> retrieve -> optional rerank -> generate -> RAGResponse
```

The pipeline records each completed stage as an `ExecutionEvent` and returns
the generated answer, final source documents, timings, metadata, and a trace
identifier together. These transport-independent models live in
`raglight.models`, so delivery and agentic layers can reuse them without
depending on the classic pipeline implementation.

Reformulation is explicit through `PipelineConfig.reformulate_queries` and
only runs for calls with conversation history. It changes the retrieval and
reranking query without replacing the original generation question.

`ConversationSession` is an application-layer wrapper around `RAGPipeline`.
The pipeline remains reusable and stateless; each session owns its identifier,
history, and lock. Completed turns are committed atomically, including streamed
turns, so failures and abandoned streams cannot leave partial history.

Directory ingestion follows the same dependency direction. `RAGPipeline`
owns deterministic filesystem traversal and per-file orchestration, resolves
processors by extension through an injected registry, and persists chunks
through `IngestionVectorStore`. It never imports a processor or storage SDK.
Concrete vector stores do not traverse directories, and failed files are
reported without hiding successful work.

`raglight.agentic.AgenticPipeline` owns a separate bounded loop:

```text
question -> decide -> optional tool -> observe -> decide -> final RAGResponse
                         ^                         |
                         +--- bounded by max_steps+
```

The runtime always registers retrieval as an agent tool. Custom tools and a
custom structural `Agent` may be injected directly. The default agent uses the
injected LLM as a JSON decision planner. Tool failures become structured
observations and failed execution events so a later decision can recover. An
unfinished loop raises a typed error after its explicit step budget.

Agentic and classic pipelines share only contracts and domain models. Neither
imports or inherits from the other, and there is no `agentic` mode flag.
Agentic sessions own history independently. Async tools use `agenerate()`;
MCP adaptation lives in `raglight.agentic.mcp` and is the only module that
loads the optional MCP SDK.

Both application layers depend only on the structural `Tracer` contract.
Nested observations are no-ops unless a tracer is injected. The optional
`LangfuseTracer` adapter lives under integrations and owns the Langfuse SDK.

## Scoped retrieval

Retrieval scopes are immutable provider-independent model values: `Eq`, `In`,
`And`, `Or`, and `MatchNone`. Classic generation, classic streaming, sessions,
and the built-in agentic retrieval tool accept the same `retrieval_filter`
value. Optional query reformulation changes only the search text; it never
changes or drops the scope. Rerankers receive only the already filtered result
set.

`None` deliberately means unrestricted retrieval for trusted standalone
library callers. `MatchNone` deliberately means no documents and is handled
without issuing a provider query. Delivery applications own authentication,
ACL evaluation, and construction of an authorized scope on every request.
Browser-supplied filter trees are not authorization decisions.

Qdrant and Chroma translate the shared tree inside their integration
boundaries. A custom vector store that receives a non-`None` scope must apply
it or raise; silently ignoring it violates the `VectorStore` contract. BM25
and hybrid modes currently reject scoped retrieval because their lexical index
does not yet retain enough metadata to enforce the same constraint safely.

The v3 agentic runtime, API server, CLI, and UI were removed during the v4
break. The current agentic runtime was rebuilt on v4 boundaries; future
delivery applications must follow the same rule rather than restore v3 code.

## Integration rules

Each integration owns:

- external SDK imports and actionable missing-dependency errors;
- provider-specific authentication and configuration;
- translation between SDK types and core models;
- deterministic provider unit tests and shared contract coverage.

Adding an integration must not add provider-name branches to the pipeline.
Built-in names are registered as lazy import references in
`raglight.integrations`; resolving one entry imports only its provider module.

## Cross-cutting invariants

- Tests are deterministic and offline by default.
- Errors identify the responsible capability and installation extra.
- Secrets never appear in logs, exceptions, fixtures, or snapshots.
- Public behavior is tested through public imports.
- Major architectural decisions are recorded in `docs/decisions/`.
