# Dependency and extras policy

The RAGLight 4 base wheel has no third-party runtime dependency. Install only
the extras required by the integrations an application imports.

| Extra | Capability | Main dependencies |
| --- | --- | --- |
| `agentic-mcp` | Adapt connected MCP client tools for Agentic RAG | Official `mcp` Python SDK |
| `langfuse` | Native classic and Agentic RAG tracing | Langfuse Python SDK |
| `openai` | OpenAI LLM and embeddings; LM Studio LLM | `langchain-openai` |
| `ollama` | Ollama LLM and embeddings | `langchain-ollama` |
| `mistral` | Mistral LLM | `langchain-mistralai` |
| `gemini` | Gemini LLM and embeddings | `langchain-google-genai` |
| `bedrock` | AWS Bedrock LLM and embeddings | `langchain-aws` |
| `huggingface` | Local embeddings and reranking | `sentence-transformers` |
| `documents` | PDF, text, code, and VLM processors | PyMuPDF, LangChain splitters |
| `chroma` | Chroma vector storage | `chromadb`, `rank-bm25` |
| `qdrant` | Qdrant vector storage | `qdrant-client`, `rank-bm25` |

Extras compose:

```bash
pip install "raglight[ollama,huggingface,qdrant]"
```

Provider modules are imported explicitly. If a package is missing, the error
names the matching installation extra. Importing `raglight`,
`raglight.contracts`, `raglight.models`, or `raglight.agentic` remains valid
with no extras. Importing `raglight.agentic.mcp` requires `agentic-mcp`, and
constructing `raglight.integrations.langfuse.LangfuseTracer` without an injected
client requires `langfuse`.

`make package-check` builds a wheel, installs it into a fresh environment with
`--no-deps`, validates core imports and metadata, and checks an actionable
missing-provider error.
