# Error handling

Errors must be actionable, safe to expose at the appropriate boundary, and
owned by the layer that can explain them.

## Rules

- Validate inputs at the earliest layer that understands the constraint.
- Preserve the original exception as the cause when translating it.
- Do not catch `Exception` merely to continue with a different provider.
- Do not leak credentials, authorization headers, local sensitive paths, or
  raw secret-bearing responses.
- A missing optional package names the exact installation extra.
- Library code raises typed exceptions; it does not print or terminate the
  process.
- Delivery layers map library errors to stable transport responses.
- Error messages explain what failed and what the caller can do next.

## Target ownership

```text
integration error -> application error -> delivery representation
```

An integration may translate an SDK exception into a RAGLight provider error.
The application may add operation context. FastAPI then maps the result to a
documented response without exposing internal tracebacks.

Every new error path needs a unit test. Publicly observable mappings also need
integration or end-to-end coverage.

Agentic tool exceptions are converted into failed `tool` execution events and
error observations so the next bounded decision can recover. They are not
silently ignored. If no final answer is produced within the configured budget,
`AgentStepLimitError` exposes the trace identifier and completed immutable
events. Invalid agent decisions fail immediately with `AgentDecisionError`.
