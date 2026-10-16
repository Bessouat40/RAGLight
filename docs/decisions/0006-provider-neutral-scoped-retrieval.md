# 0006: Express retrieval scope with a provider-neutral filter tree

- Status: accepted
- Date: 2026-10-16

## Context

Applications need to constrain retrieval by tenant, workspace, file, or active
document version. Qdrant and Chroma expose different native filter types, and
the original public `VectorStore` contract had no filtering argument. Passing
provider dictionaries through the pipeline would couple core to integrations
and make security behavior vary silently by backend.

Agentic retrieval adds another risk: an agent-controlled tool call must not be
able to omit or replace the authorization scope selected by its application.
The existing BM25 index stores text without metadata, so it cannot enforce the
same scope safely.

## Decision

Core owns immutable `Eq`, `In`, `And`, `Or`, and `MatchNone` retrieval-filter
models. `RAGPipeline`, conversation sessions, `AgenticPipeline`, agentic
sessions, and `VectorStore.similarity_search()` accept the same optional
`retrieval_filter` keyword.

`None` means unrestricted retrieval for trusted standalone use. `MatchNone`
means no documents and short-circuits without a backend query. Classic query
reformulation preserves the filter. Agentic execution binds the filter outside
the model-visible tool arguments and supplies it to every built-in retrieval
invocation.

Integrations translate the tree into native SDK values. Unsupported scoped
search modes fail closed. BM25 and hybrid retrieval reject scopes until their
lexical index preserves metadata and can filter candidates before ranking.
Authentication and ACL resolution remain application responsibilities.

## Consequences

- Applications can derive one authorization scope independently of the chosen
  semantic provider.
- Provider SDK types do not cross core contracts.
- A custom adapter may continue serving unrestricted calls, but must update
  its signature and either apply or reject non-`None` scopes.
- Qdrant and Chroma share executable filter-parity tests.
- Lexical and hybrid scoped search require a separate metadata-preserving
  implementation before they are safe for multi-user applications.

## Alternatives considered

Accepting arbitrary provider dictionaries was rejected because it leaks SDK
semantics into core and prevents portable contract tests. Filtering the top-k
results after retrieval was rejected because unauthorized candidates can
displace allowed documents and may already have crossed the retrieval
boundary. Storing a mutable filter on a shared pipeline or retrieval tool was
rejected because concurrent requests could leak scopes across users.
