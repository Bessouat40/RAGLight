# 0005: Keep Agentic RAG separate and bounded

- Status: accepted
- Date: 2026-10-16

## Context

Classic RAG has a deterministic retrieval, optional reranking, and generation
flow. Agentic RAG instead selects tools iteratively, observes failures, and may
not terminate unless the application enforces a budget. RAGLight v3 mixed this
behavior with legacy configuration and framework-specific implementation.

## Decision

RAGLight v4 exposes `raglight.agentic.AgenticPipeline` as a sibling of
`RAGPipeline`. The classes do not inherit from one another, and classic RAG has
no agentic mode flag.

The agentic runtime owns structural `Agent` and `AgentTool` contracts, a tool
registry, the built-in retrieval tool, the LLM-backed JSON planner, bounded
execution, and isolated sessions. Every decision consumes one configured
step. Tool failures become observations; exhausting the budget raises a typed
error with completed events.

Custom agents are injected directly. This permits framework adapters without
making any framework a core dependency. MCP tools are asynchronous adapters
over a caller-managed official client and live behind `agentic-mcp`.

## Consequences

- Classic imports and installations do not load agent or MCP dependencies.
- Retrieval and optional reranking reuse existing v4 contracts.
- Agent behavior is replaceable without inheritance or provider branches.
- Synchronous and asynchronous tool execution share one response model.
- Applications must choose an explicit step budget and own MCP connection
  lifecycle, authorization, and human-approval policy.
