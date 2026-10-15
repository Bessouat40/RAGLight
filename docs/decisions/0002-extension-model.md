# 0002: Extend RAGLight through contracts and dependency injection

- Status: proposed
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

The final method signatures and compatibility guarantees will be accepted as
part of issue #157.

## Consequences

- Custom components work without editing the builder.
- Registries remain a convenience rather than a privileged internal path.
- Contract tests become part of integration acceptance.
- Existing provider classes may require adapters during migration.

## Alternatives considered

Continuing to add `if`/`elif` branches was rejected because it couples every
integration to framework releases. Runtime duck typing without documented
contracts was rejected because failures would surface too late.

