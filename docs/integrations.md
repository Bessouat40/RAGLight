# Integration guide

An integration connects a core RAGLight contract to an external provider or
SDK. New integrations must not require changes to unrelated core behavior.

## Before implementation

- Identify the core contract implemented.
- Confirm the feature cannot be expressed by configuring an existing adapter.
- Choose a logical optional extra and installation command.
- Record authentication, network, retry, timeout, and streaming semantics.
- Define behavior for an unavailable package and invalid credentials.

## TDD sequence

1. Add the implementation to the shared contract-test matrix as a failing case.
2. Add provider-specific unit tests with the external SDK mocked at its edge.
3. Implement loading and translation behavior.
4. Add public-import and configuration coverage.
5. Add an opt-in live smoke test only when it provides unique value.

## Required behavior

Every integration must:

- implement the complete extension contract;
- receive configuration explicitly;
- preserve core input and output semantics;
- support deterministic unit tests without network access;
- raise an actionable missing-dependency error such as
  `pip install "raglight[provider]"`;
- avoid importing its SDK when the integration is not selected;
- avoid logging credentials, prompts marked sensitive, or raw secret-bearing
  provider responses;
- document limitations and provider-specific behavior.

Built-in integrations register a lazy public import path in
`raglight.integrations.default_registries`. Do not import the provider module
while populating the registry. Third-party components may call `register(...)`
when they need name-based configuration; direct injection remains available
without registration.

Document processor registry names used by directory ingestion are normalized
file extensions without a leading dot, such as `txt`, `md`, `pdf`, or `py`.
Registering a custom extension is enough for `RAGPipeline.ingest(...)` to use
it; the pipeline must not be modified. Processor exceptions should propagate
to the orchestrator, which records an actionable per-file failure.

The extension contracts are structural. A provider class must not inherit from
a protocol merely to satisfy RAGLight; matching the documented signature is
sufficient. Add every built-in implementation to the shared matrix in
`tests/contracts/test_extension_contracts.py`, then keep provider-specific
translation behavior in its own unit tests.

Agent tools use the same structural approach inside `raglight.agentic`. MCP is
different from a provider integration because the caller owns a live client
connection. `raglight.agentic.mcp.MCPTool` adapts one tool on that initialized
client and never owns transport startup or credentials. Keep the official SDK
behind the `agentic-mcp` extra and use `AgenticPipeline.agenerate(...)` while
the client connection is active.

Vector-store integrations translate the provider-independent
`RetrievalFilter` tree at their SDK boundary. Semantic stores must implement
the documented `Eq`, `In`, `And`, `Or`, and `MatchNone` behavior consistently.
`MatchNone` must short-circuit without a provider query. If a search mode
cannot apply a requested filter before top-k selection, it must raise an
actionable error instead of ignoring the scope or filtering results afterward.
Shared integration tests must prove parity with another built-in store and use
documents whose relevance would expose an accidental unrestricted query.

## Pull request checklist

- [ ] Core contract identified.
- [ ] Shared contract tests pass.
- [ ] Provider-specific success and error paths are unit tested.
- [ ] Streaming behavior is tested when supported.
- [ ] Optional dependency and lock file are updated.
- [ ] Base `import raglight` works without the provider SDK.
- [ ] Missing dependency message names the correct extra.
- [ ] Public import and configuration path is integration tested.
- [ ] Retrieval-filter operators pass the shared parity suite, when applicable.
- [ ] Built-in registry entry is lazy and covered by the registry matrix.
- [ ] Documentation uses public imports only.
- [ ] No real credentials or required network calls are present.

Provider-name conditionals do not belong in application code. Registries may
offer name-based construction, but direct component injection remains the
extension mechanism.
