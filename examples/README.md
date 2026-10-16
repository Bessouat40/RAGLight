# RAGLight v4 examples

These small programs use only the documented RAGLight v4 APIs. Run every
command from the repository root. Provider examples make real calls; the two
examples under `extending/` and the custom-agent example are fully offline.

## Classic RAG

### Ollama

Install the integrations and pull the default models:

```bash
uv sync --extra ollama --extra chroma
ollama pull nomic-embed-text:latest
ollama pull qwen3:0.6b
uv run python examples/classic/ollama.py
```

The script uses one inline document and an ephemeral Chroma database. Override
the question, model names, or Ollama URL with `--help`.

### OpenAI

Set `OPENAI_API_KEY` in your environment; never put it in source code.

```bash
uv sync --extra openai --extra chroma
export OPENAI_API_KEY="your-key"
uv run python examples/classic/openai.py
```

Change models with `--chat-model` and `--embedding-model` when needed.

### OpenAI-compatible endpoint

The custom endpoint must expose compatible chat-completions and embeddings
APIs. Model identifiers depend on the server, so both are explicit arguments.

```bash
uv sync --extra openai --extra chroma
uv run python examples/classic/openai_compatible_endpoint.py \
  --base-url http://localhost:1234/v1 \
  --chat-model your-chat-model \
  --embedding-model your-embedding-model
```

Pass `--api-key` or set `OPENAI_API_KEY` when the endpoint requires one.

### Recursively ingest a directory

This workflow supports registered text, Markdown, HTML, PDF, and source-code
processors. Its Chroma database is temporary, so the script demonstrates a
complete ingest-and-query run without leaving local data behind.

```bash
uv sync --extra ollama --extra chroma --extra documents
uv run python examples/classic/directory_ingestion.py ./knowledge \
  --question "What are the main topics in these documents?"
```

## Agentic RAG

Agentic RAG is a separate API with a bounded number of decisions. Small local
models may occasionally produce invalid planning JSON; choose a stronger
instruction-following Ollama model with `--chat-model` if necessary.

### Ollama planner

```bash
uv sync --extra ollama --extra chroma
uv run python examples/agentic/ollama.py --max-steps 4
```

### Custom agent

This example is offline and shows the complete structural `decide(context)`
contract without an agent framework or inheritance.

```bash
uv run python examples/agentic/custom_agent.py
```

### MCP tool

The caller owns the MCP connection. This example connects to an existing
streamable-HTTP MCP server, discovers the requested tool, keeps the client open
during agent execution, and uses Ollama for planning.

```bash
uv sync --extra agentic-mcp --extra ollama --extra chroma
uv run python examples/agentic/mcp.py \
  --server-url http://localhost:8000/mcp \
  --tool search \
  --question "Use search when it helps answer the question."
```

Use only MCP servers and tools you trust. Authentication and server startup
are intentionally outside RAGLight's connection-lifecycle boundary.

## Extending RAGLight

Neither example needs an optional integration extra:

```bash
uv run python examples/extending/custom_components.py
uv run python examples/extending/registry_configuration.py
```

- `extending/custom_components.py` injects plain Python implementations
  directly.
- `extending/registry_configuration.py` registers named factories and builds a
  pipeline from serializable configuration.

## Example index

| Example | External requirement |
| --- | --- |
| `classic/ollama.py` | Running Ollama and pulled models |
| `classic/openai.py` | Network access and `OPENAI_API_KEY` |
| `classic/openai_compatible_endpoint.py` | Compatible chat and embedding server |
| `classic/directory_ingestion.py` | Running Ollama and a readable directory |
| `agentic/ollama.py` | Running Ollama and pulled models |
| `agentic/custom_agent.py` | None |
| `agentic/mcp.py` | Running Ollama and a connected MCP server |
| `extending/custom_components.py` | None |
| `extending/registry_configuration.py` | None |
