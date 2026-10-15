# 0003: Expose one small, layered public API in v4

- Status: accepted
- Date: 2026-10-15

## Context

RAGLight v3 exposes multiple ways to construct the same workflow: `Builder`
creates the low-level `RAG` object, `RAGPipeline` combines two configuration
objects, and `AgenticRAGPipeline` introduces another configuration model.
Concrete providers and global settings are also re-exported from the package
root. This makes internal modules appear stable and causes optional provider
dependencies to load during common imports.

The v4 extension, lightweight-installation, structured-response, and session
issues all need one API destination before packages are moved.

## Decision

RAGLight v4 has one standard pipeline class: `raglight.RAGPipeline`.

The public API is layered by purpose:

1. `raglight` exposes only `RAGPipeline` and `PipelineConfig`.
2. `raglight.models` exposes transport-independent domain values.
3. `raglight.contracts` exposes structural component contracts.
4. `raglight.integrations.<provider>` exposes optional concrete adapters.
5. Optional agentic, observability, and delivery capabilities use dedicated
   namespaces and extras.

Direct component injection is canonical. `RAGPipeline.from_config(...)`
provides the registry-backed, serializable convenience path and receives its
registries explicitly. `Builder`, the
low-level execution graph, global provider settings, and concrete integrations
are not root exports.

The standard pipeline is stateless. `generate()` returns the shared
`RAGResponse` domain model with structured execution events. Isolated sessions
are finalized separately.

Agentic execution is exposed separately as
`raglight.agentic.AgenticPipeline`. It shares contracts and domain models with
`RAGPipeline` but does not inherit from it. Standard RAG has no agentic mode
flag, and its installation does not require agent or MCP dependencies.

## Consequences

- Application code has one obvious entry point.
- Core imports can remain independent of provider SDKs.
- Custom components and built-in providers use the same pipeline path.
- Provider-specific options remain at integration boundaries.
- v3 deep imports and builder-based examples require a migration guide.
- Agentic-specific configuration remains out of the standard pipeline while
  common components and response types remain reusable.
- Internal compatibility shims may be necessary while the migration is in
  progress, but they are removed before the v4 release.

## Alternatives considered

Keeping both `Builder` and `RAGPipeline` public was rejected because they expose
two composition models and make extension behavior ambiguous.

Making provider strings the primary constructor API was rejected because each
new integration would continue to influence the application layer.

Re-exporting every useful class from `raglight` was rejected because it makes
the root import unstable and risks importing optional dependencies.

Using one pipeline class with an `agentic` flag was rejected because standard
RAG and agentic loops have different execution, tool, failure, and dependency
semantics. Public inheritance between the two pipelines was rejected because
their commonality is component composition rather than substitutable behavior.
