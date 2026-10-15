# 0001: Dependency boundaries point toward the core

- Status: accepted
- Date: 2026-10-15

## Context

RAGLight v3 mixes contracts, concrete providers, orchestration, and delivery
concerns closely enough that adding integrations can modify framework internals
and importing the package can load unrelated dependencies.

## Decision

RAGLight v4 uses four conceptual layers: core, application, integrations, and
delivery. Core owns contracts and transport-independent models. Application
coordinates use cases. Integrations implement core contracts. Delivery adapts
HTTP, CLI, and UI interactions.

Dependencies point toward core. Architecture boundaries are enforced by
executable checks rather than prose alone.

## Consequences

- Custom components can be injected without editing the framework.
- Optional integrations can be packaged independently.
- Server and frontend work can reuse application behavior.
- Removed v3 packages are not kept as compatibility layers.
- New v4 code must follow the boundary immediately.

## Alternatives considered

Keeping provider selection in a central builder was rejected because every new
provider would continue to modify application internals.
