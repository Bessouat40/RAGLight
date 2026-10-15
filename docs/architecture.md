# Architecture

This document separates the architecture that exists in v3 from the boundaries
that v4 must enforce. The distinction is intentional: target rules are not a
claim that the migration is already complete.

## Current v3 architecture

RAGLight currently uses abstract base classes for LLMs, embeddings, vector
stores, and document processors. Pipeline construction and concrete-provider
selection are both concentrated in the builder. Provider implementations and
their contracts share packages, while the CLI, FastAPI server, and Streamlit UI
call into the library.

The main data flows are:

```text
source -> document processor -> embeddings -> vector store

question -> retrieval -> optional reranking -> generation -> answer
```

Important v3 entry points include `RAGPipeline`, `Builder`, `RAG`, and
`AgenticRAGPipeline`. The v4 public API work will decide which names remain
public; their existence today does not guarantee future stability.

## Target v4 layers

```text
delivery       HTTP API, CLI, frontend
    |
application    pipeline use cases, sessions, ingestion orchestration
    |
core           contracts, domain models, response and event models
    ^
integrations   concrete LLM, embedding, vector-store and processor adapters
```

Dependencies point toward the core. Delivery layers invoke application use
cases. Integrations implement core contracts and are injected into the
application layer.

The target package ownership is:

```text
raglight.contracts       stable component protocols
raglight.models          transport-independent values
raglight.pipeline        application orchestration (internal package)
raglight.integrations    optional provider adapters and registries
raglight.agentic         optional agentic application capability
raglight.server          optional HTTP delivery capability
raglight.cli             optional command-line delivery capability
raglight.ui              optional user-interface delivery capability
```

`raglight.RAGPipeline` is a public facade over the internal application
package. Package layout and public API are related but not identical: internal
application modules do not become public merely because the facade delegates
to them.

### Core

The core owns stable contracts and transport-independent models. It must not
import concrete providers, FastAPI, Streamlit, or CLI packages. Importing the
core must not require an optional integration dependency.

### Application

The application layer coordinates retrieval, reranking, generation, ingestion,
and sessions. It depends on contracts rather than concrete providers.
Conversation state is supplied by a caller or session and is never global
pipeline state.

### Integrations

Each integration implements a core contract. It owns translation to an
external SDK, dependency checks, provider-specific configuration, and contract
tests. Integrations must not change the core to add provider-specific branches.

### Delivery

FastAPI, CLI, and UI code translate transport or interaction details to
application calls. They may handle authentication, request validation, and
presentation, but they do not implement retrieval or generation decisions.

## Dependency rules

Immediately enforced:

- code outside delivery packages cannot import `raglight.api`,
  `raglight.cli`, or `raglight.ui`;
- code outside delivery packages cannot import FastAPI or Streamlit.

Target-v4 rules introduced as the migration progresses:

- core contracts do not import concrete integrations;
- optional providers are not imported when the base package loads;
- REST models do not become public domain models;
- provider registration is extensible without editing application internals.

The executable checks are in `scripts/check_architecture.py` and
`tests/architecture/`. Add a guardrail when a boundary becomes enforceable; do
not rely on prose after the migration lands.

## Cross-cutting invariants

- Errors are actionable and identify the responsible layer.
- Structured responses and events are transport independent.
- Tests are deterministic and offline by default.
- Configuration does not silently select a different provider.
- Secrets never appear in logs, exceptions, fixtures, or snapshots.
- Public API behavior is covered through public imports.

Major decisions belong in `docs/decisions/` and include context, decision,
consequences, and status.
