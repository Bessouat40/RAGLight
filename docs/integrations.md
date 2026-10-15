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

The extension contracts are structural. A provider class must not inherit from
a protocol merely to satisfy RAGLight; matching the documented signature is
sufficient. Add every built-in implementation to the shared matrix in
`tests/contracts/test_extension_contracts.py`, then keep provider-specific
translation behavior in its own unit tests.

## Pull request checklist

- [ ] Core contract identified.
- [ ] Shared contract tests pass.
- [ ] Provider-specific success and error paths are unit tested.
- [ ] Streaming behavior is tested when supported.
- [ ] Optional dependency and lock file are updated.
- [ ] Base `import raglight` works without the provider SDK.
- [ ] Missing dependency message names the correct extra.
- [ ] Public import and configuration path is integration tested.
- [ ] Documentation uses public imports only.
- [ ] No real credentials or required network calls are present.

Provider-name conditionals do not belong in application code. Registries may
offer name-based construction, but direct component injection remains the
extension mechanism.
