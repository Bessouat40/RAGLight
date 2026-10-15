# Development

## Requirements

- Python 3.11 or newer, matching `pyproject.toml`
- `uv`
- Git

Install the development environment:

```bash
make install-dev
```

The default test suite is offline. Do not configure provider credentials merely
to run normal development checks.

## Focused feedback

Run one existing unittest module:

```bash
PYTHONPATH=src uv run python -m unittest -v tests.tests_rag.test_rag
```

Run one test method:

```bash
PYTHONPATH=src uv run python -m unittest -v \
  tests.tests_rag.test_rag.TestRAGGenerate.test_generate
```

Use `make test-unit`, `make test-integration`, and `make test-e2e` for the
standard tiers. During the v3-to-v4 migration, the historical test directories
are explicitly listed in the unit command; new cross-component and workflow
tests go in their explicitly named directories. When adding a new unit-test
package, add it to `test-unit` in the `Makefile` until issue #161 consolidates
the legacy layout.

## Validation

```bash
make format-check       # Black, check only
make lint               # Ruff correctness rules
make typecheck          # currently enforced typed tooling scope
make architecture-check # dependency boundaries
make package-check      # build and inspect/install the wheel
make check              # complete local gate
```

The typing and lint scopes start from a baseline the v3 tree can satisfy. They
must become stricter as v4 modules are introduced. A new v4 module must be fully
typed even if a legacy neighbor is not.

## Test environment rules

- Patch provider clients at the integration boundary.
- Use fake implementations for application tests.
- Never call a live provider from the default suite.
- Never use sleeps to coordinate asynchronous behavior.
- Keep fixtures small, deterministic, and free of personal or secret data.
- Use temporary directories supplied by the test framework.

## Changing dependencies

Explain why the dependency belongs in the base package or a named extra. After
changing packaging, run `make package-check` and verify the relevant minimal or
extra installation. Issue #156 will establish the complete extras matrix.
