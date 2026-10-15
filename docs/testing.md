# Testing strategy

RAGLight uses test-driven development for behavior changes. Tests document
observable contracts and protect refactoring; they should not merely reproduce
the implementation.

## Red, green, refactor

1. Write a test for one behavior.
2. Run it and confirm it fails for the intended reason.
3. Add the smallest implementation that makes it pass.
4. Improve names and structure while the test remains green.
5. Add the next behavior as another failing example.

For a bug, first reproduce the bug in a regression test. A test that already
passes before the fix does not prove the regression.

## Test pyramid

### Unit tests

Required for every behavior change. Unit tests isolate one unit, are fast, and
replace external clients with fakes or mocks. Examples include response-model
validation, registry behavior, configuration validation, and session history.

### Contract tests

Required for every implementation of an extension contract. A shared suite
must exercise the same behavior for every LLM, embeddings implementation, and
vector store. Provider-specific tests supplement rather than replace the shared
contract.

### Integration tests

Required when behavior crosses component boundaries. Prefer an in-memory or
fake implementation. Examples include pipeline plus fake LLM and vector store,
or ingestion plus an in-memory store.

### End-to-end tests

Required for new or changed critical public workflows. Exercise only public
entry points. Representative workflows are:

- ingest documents, retrieve context, and generate a structured response;
- create two sessions and prove their histories are isolated;
- upload, ingest, stream an answer, and inspect sources through the server;
- later, complete the same workflow through the dedicated frontend.

End-to-end does not mean live cloud access. Default E2E tests use deterministic
fakes and an in-process HTTP client. Live-provider smoke tests are opt-in and
must be labeled, documented, and excluded from required offline CI.

## Choosing coverage

| Change | Unit | Contract | Integration | E2E |
| --- | --- | --- | --- | --- |
| Pure internal refactor | existing | if contract touched | if boundary touched | no |
| New provider | yes | required | registration/loading | optional live smoke |
| Pipeline behavior | required | as applicable | required | public workflow |
| HTTP endpoint | validation | no | application mapping | required |
| Bug fix | regression | if contract bug | if cross-component | if public regression |
| Documentation only | not applicable | no | no | no |

## Test quality

- Name tests after behavior and outcome.
- Assert public outputs or state, not private call order, unless interaction is
  the contract.
- Keep one reason to fail per test.
- Avoid shared mutable state and order dependence.
- Test errors and boundary cases, not only happy paths.
- Do not weaken an assertion to make an implementation pass.

Coverage is a diagnostic, not a substitute for behavioral tests. CI may enforce
changed-code coverage once a trustworthy baseline exists; it must not reward
tests that execute code without meaningful assertions.

