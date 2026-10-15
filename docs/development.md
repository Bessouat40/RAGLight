# Development

## Setup

RAGLight requires Python 3.11 or newer, `uv`, and Git.

```bash
make install-dev
```

The required suite is offline; normal development must not require provider
credentials or network calls.

## TDD workflow

1. Add a test that fails for the intended reason.
2. Implement the smallest behavior that makes it pass.
3. Refactor while keeping the suite green.
4. Add integration or end-to-end coverage when a public workflow crosses
   component boundaries.

Run a focused test during the red/green loop:

```bash
PYTHONPATH=src uv run python -m unittest -v \
  tests.tests_rag.test_public_api.PublicAPITests
```

## Validation

```bash
make test-unit
make test-integration
make test-e2e
make architecture-check
make package-check
make check
```

`make check` is the required local gate. It includes formatting, linting, type
checking, architecture rules, every test tier, and isolated wheel validation.

## Contribution rules

- Test public behavior, not private implementation details.
- Mock external SDKs at the integration boundary.
- Use temporary directories and deterministic fixtures.
- Put dependencies in the narrowest matching extra and update `uv.lock`.
- Do not introduce global provider selection or credentials.
- Do not make the pipeline import a concrete integration.
- Document new public names and migration impact in the same change.
