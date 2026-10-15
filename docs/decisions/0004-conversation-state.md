# 0004: Scope conversation state to explicit sessions

- Status: accepted
- Date: 2026-10-15

## Context

A configured RAG pipeline is expensive and useful across many requests, while
conversation history belongs to one user interaction. Storing messages on a
shared pipeline would mix concurrent users and make stateless application
requests impossible. Streaming adds another boundary: a partial answer must not
become durable conversation state.

## Decision

`RAGPipeline` remains stateless. Its `generate()` and `stream()` methods accept
explicit history for callers that manage state themselves.

`RAGPipeline.create_session()` creates a `ConversationSession` for callers that
want managed history. A session owns its identifier, copied history, and a
per-session lock. It appends the user and assistant messages only after a
generation succeeds or a stream is fully consumed. Failed or abandoned turns
leave history unchanged. History exposed to callers is a detached snapshot.

Sessions may share one pipeline and its configured integrations. They never
share conversation history or synchronization state.

## Consequences

- One pipeline can serve concurrent users without cross-session message leaks.
- Stateless and stateful calling styles use the same retrieval and generation
  implementation.
- Applications decide session lifetime and persistence; the core keeps no
  global session registry.
- A streamed answer is not committed until its iterator is exhausted.
- Callers that abandon a stream must start a new turn because partial text is
  intentionally absent from committed history.

## Alternatives considered

Keeping history on `RAGPipeline` was rejected because pipeline reuse would
couple unrelated users.

A global session registry was rejected because persistence, eviction, and
distributed ownership belong to delivery applications.

Committing each streaming chunk was rejected because failed or cancelled
streams would create incomplete assistant messages.
