# Dependency and extras policy

RAGLight v4 has a dependency-free base installation. The base wheel exposes
`PipelineConfig` and `RAGPipeline`; integrations and delivery applications are
installed only when selected.

## Installation matrix

| Extra | Capability | Main dependencies |
| --- | --- | --- |
| `openai` | OpenAI, vLLM, and LM Studio models and embeddings | `langchain-openai` |
| `ollama` | Ollama models and embeddings | `langchain-ollama` |
| `mistral` | Mistral models | `langchain-mistralai` |
| `gemini` | Google Gemini models and embeddings | `langchain-google-genai` |
| `bedrock` | AWS Bedrock models and embeddings | `langchain-aws` |
| `huggingface` | Local embeddings and reranking | `sentence-transformers` |
| `chroma` | Chroma vector storage | `chromadb` plus ingestion support |
| `qdrant` | Qdrant vector storage | `qdrant-client` plus ingestion support |
| `documents` | PDF, text, code, and VLM ingestion | PyMuPDF and LangChain splitters |
| `langfuse` | Langfuse tracing | `langfuse` |
| `agentic` | Legacy agent orchestration and MCP tools | LangChain agents and MCP adapters |
| `server` | FastAPI delivery application | FastAPI and Uvicorn |
| `cli` | Interactive command-line application | Typer, Rich, NLTK, and legacy orchestration |
| `ui` | Streamlit user interface | Streamlit and Requests |
| `legacy` | Shared v3 orchestration compatibility | LangGraph, ingestion, and hybrid retrieval |

Extras compose. A typical v3 local setup using Ollama, Hugging Face embeddings,
and Qdrant is installed with:

```bash
pip install "raglight[legacy,ollama,huggingface,qdrant]"
```

Add `agentic` only for agent workflows, and add `cli`, `server`, or `ui` only
for the delivery application you run. The standard v4 pipeline does not require
the `agentic` extra.

## Audit decisions

Every former mandatory dependency was assigned to the narrowest owning extra.
Direct requirements on `mistralai`, `ollama`, `google-genai`, and `transformers`
were removed because RAGLight does not import them directly; their LangChain or
Sentence Transformers adapters declare the SDK versions they support. Requests
was added explicitly to `ui` because the Streamlit application imports it.

The compatibility extras currently repeat a few shared requirements. This is
intentional: installing a documented extra must produce a usable feature, not
an incomplete environment that relies on an unrelated extra being present.
Issue #157 will move provider contracts and adapters into their final v4
boundaries; it may reduce this legacy overlap without changing the lightweight
base guarantee.

## Missing dependencies

Provider modules are loaded only when the corresponding provider is selected.
When an optional package is absent, RAGLight reports the exact installation
command, for example:

```text
Ollama support requires optional dependencies. Install them with: pip install 'raglight[ollama]'
```

The `raglight` console script follows the same rule and points to
`raglight[cli]` instead of failing with an unexplained import error.

## Validation

`make package-check` builds the wheel, installs it into a fresh environment
with `--no-deps`, checks its metadata for mandatory requirements, imports the
public core API, and verifies the provider and CLI missing-dependency messages.
The full test environment continues to install every extra until issue #162
introduces an extra-specific CI matrix.
