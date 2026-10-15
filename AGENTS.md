# RAGLight contributor and agent guide

This file is the authoritative starting point for humans and AI coding agents
working in this repository. Tool-specific instruction files must link here and
must not duplicate these rules.

RAGLight is a Python library for building retrieval-augmented generation
pipelines. Version 4 aims to keep the core small, make integrations optional,
and expose stable extension contracts without coupling the library to its CLI,
HTTP server, or user interface.

## Start here

Read the documents relevant to the change before editing code:

| Document | Purpose |
| --- | --- |
| `docs/architecture.md` | Current architecture, v4 target layers, and dependency rules |
| `docs/development.md` | Environment setup and validation commands |
| `docs/dependencies.md` | Optional extras, dependency ownership, and package audit |
| `docs/testing.md` | TDD workflow and test pyramid |
| `docs/integrations.md` | Checklist for adding or changing a provider |
| `docs/public-api.md` | Public API policy and compatibility expectations |
| `docs/migration-v3-to-v4.md` | Authoritative migration map from legacy APIs |
| `docs/error-handling.md` | Error ownership and actionable failures |
| `docs/releasing.md` | Package and release validation |
| `docs/decisions/` | Architecture decision records |

Documentation explicitly distinguishes **Current v3**, **Target v4**, and
**Migration rule** statements. Do not describe a target rule as already
implemented.

## Repository map

```text
src/raglight/
  models/                shared data models
  llm/                   LLM contracts and provider implementations
  embeddings/            embedding contracts and providers
  vectorstore/           vector-store contracts and providers
  document_processing/   ingestion processors
  rag/                    pipeline construction and orchestration
  api/                    FastAPI delivery layer
  cli/                    command-line delivery layer
  ui/                     Streamlit user interface
tests/                    existing unit and component tests
tests/architecture/       executable dependency guardrails
tests/integration/        cross-component tests
tests/e2e/                public workflow tests
```

The v3 tree does not yet fully match the target v4 layers. Preserve current
behavior while moving it toward the boundaries in `docs/architecture.md`.

## Non-negotiable engineering rules

1. Use test-driven development for behavior changes: red, green, refactor.
2. Every behavior change has at least a unit test.
3. Every bug fix has a regression test that fails without the fix.
4. Tests run without network access, credentials, or external services unless
   explicitly marked as opt-in live tests.
5. Core code does not depend on `raglight.api`, `raglight.cli`,
   `raglight.ui`, FastAPI, or Streamlit.
6. Core extension contracts never depend on concrete provider implementations.
7. Optional packages are imported only within their integration boundary and
   missing dependencies produce an actionable installation message.
8. HTTP and UI layers translate inputs and outputs; they do not own RAG
   business logic.
9. Conversation history belongs to a caller or session, never to a globally
   shared pipeline.
10. Public API changes require tests, documentation, and migration notes.
11. CI validates source code and never rewrites or commits it.
12. Do not mix unrelated cleanup with a focused behavior change.

Some target-v4 rules expose existing v3 debt. Do not conceal that debt with a
weak check. Record the exception, keep the guardrail from getting worse, and
remove the exception in the issue that completes the migration.

## TDD workflow

For each observable behavior:

1. Write the acceptance example in user-visible terms.
2. Identify the layer that owns the behavior.
3. Add the smallest test that expresses it.
4. Run the test and confirm it fails for the expected reason.
5. Implement the minimum code needed to pass.
6. Refactor while keeping the focused test green.
7. Add contract, integration, or end-to-end coverage required by
   `docs/testing.md`.
8. Run the focused suite, then the complete relevant validation.
9. Update documentation in the same change.

Do not write production code first and add superficial tests afterward. TDD is
a development discipline; CI cannot reconstruct the red/green history, so code
review must verify that tests describe behavior rather than implementation.

## Adding a feature

Before implementation, answer these questions in the issue or pull request:

- What observable behavior changes?
- Which layer owns it?
- Is the public Python API affected?
- Does it add or change an optional dependency?
- Which unit, contract, integration, and end-to-end tests are required?
- Which documentation and migration notes must change?

Prefer dependency injection and stable contracts over provider-name branches.
Keep transport-specific models at the repository edge. See
`docs/integrations.md` for provider work.

## Validation commands

Use the smallest command during development and the complete gate before
finishing:

```bash
make test-unit
make test-integration
make test-e2e
make architecture-check
make format-check
make lint
make typecheck
make package-check
make check
```

See `docs/development.md` for focused test commands and environment details.

## Definition of done

A behavior change is complete only when:

- acceptance criteria are represented by tests;
- unit tests cover the new behavior;
- bug fixes include regression tests;
- implementations of extension contracts pass shared contract tests;
- affected component interactions have integration coverage;
- affected public workflows have end-to-end coverage where appropriate;
- architecture, formatting, lint, typing, and package checks pass at their
  currently enforced scope;
- documentation and migration guidance are current;
- default tests need no secrets, network, or running external service;
- the base package remains importable without unrelated optional extras.

Documentation-only changes may state why runtime tests are not applicable, but
documentation checks and relevant static validation still apply.

## Working safely in an existing worktree

- Inspect `git status` before editing.
- Treat existing modifications as user-owned unless explicitly told otherwise.
- Do not discard, rewrite, stage, or commit unrelated changes.
- Re-read shared files immediately before editing them.
- Keep commits focused and make generated changes explicit.
- Never place credentials, tokens, local databases, or provider responses in
  tests or fixtures.
- If another change prevents validation, report the exact command and failure;
  do not silently modify unrelated code to make the tree green.
