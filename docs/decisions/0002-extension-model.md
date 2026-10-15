# 0002: Extend RAGLight through contracts and dependency injection

- Status: accepted
- Date: 2026-10-15

## Context

Users need custom LLM, embeddings, and vector-store implementations without
forking RAGLight. Built-in provider strings remain useful for configuration but
must not define the extension model.

## Decision

The primary extension mechanism is passing objects that implement stable core
contracts. Registries provide optional name-based construction for built-in and
third-party integrations. Every implementation passes a shared contract-test
suite.

The public contracts are Python structural protocols. Implementations provide
the required methods but do not inherit from the protocols. The v4 signatures
are documented in `docs/public-api.md`; `raglight.models.Document` is the value
exchanged by retrieval and reranking contracts.

The v3 abstract base classes and cross-encoder adapter are removed. Built-in
components conform structurally to the same protocols as custom components.

## Consequences

- Custom components work without editing the builder.
- Registries remain a convenience rather than a privileged internal path.
- Contract tests become part of integration acceptance.
- Provider adapters can evolve internally without changing core contracts.

## Alternatives considered

Continuing to add `if`/`elif` branches was rejected because it couples every
integration to framework releases. Runtime duck typing without documented
contracts was rejected because failures would surface too late.
