# Public API policy

## Current status

RAGLight v3 exposes classes from both top-level and deep module paths. Some
internal modules are used by examples and may appear public even though their
stability was never explicitly defined.

Issue #155 will define the authoritative v4 public surface. Until then, do not
claim that a deep import is a stable v4 API.

## Target v4 policy

- Common usage relies on documented top-level imports.
- Extension contracts have explicit documented import paths.
- Internal packages are not required for normal use.
- Public inputs and outputs are transport independent.
- Public behavior is tested through public imports.
- Breaking changes include migration notes and release documentation.
- Deprecations state the replacement and removal version.

Adding a name to `raglight.__init__` is a compatibility commitment, not a
convenience-only change. Review ownership, typing, documentation, dependency
impact, and tests before exporting it.

## Review questions

- Can the behavior be expressed through an existing public abstraction?
- Does this expose a concrete provider where a contract should be exposed?
- Does importing the symbol load an optional dependency?
- Is the model useful outside HTTP, CLI, or UI delivery?
- Can the API evolve without adding provider-specific arguments?
- Is the behavior covered using only public imports?

