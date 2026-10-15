# RAGLight

RAGLight 4 is a small, typed Python toolkit for composing retrieval-augmented
generation pipelines. The core package has no third-party runtime dependency:
applications inject an LLM, an embedding model, a vector store, and optionally
a reranker through public protocols.

> RAGLight 4 is a breaking release. The v3 builder, global settings, agentic
> runtime, API server, CLI, and UI have been removed instead of being carried as
> compatibility layers.

## Requirements

- Python 3.11 or newer
- One optional integration extra when using a built-in provider

```bash
pip install raglight
pip install "raglight[ollama,chroma]"
```

Available extras are `openai`, `ollama`, `mistral`, `gemini`, `bedrock`,
`huggingface`, `documents`, `chroma`, and `qdrant`.

## Public API

The stable core is deliberately narrow:

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.contracts import (
    DocumentProcessor,
    EmbeddingModel,
    LLM,
    Reranker,
    VectorStore,
)
from raglight.models import Document
```

The contracts are structural protocols. Custom components do not inherit from
RAGLight classes and do not need registry entries.

```python
from raglight import PipelineConfig, RAGPipeline
from raglight.models import Document


class MyLLM:
    def generate(self, input: dict[str, object]) -> str:
        contexts = input.get("contexts", [])
        return f"Answer based on {len(contexts)} documents"


class MyEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class MyVectorStore:
    def similarity_search(
        self, question: str, k: int = 5, filter: dict[str, object] | None = None
    ) -> list[Document]:
        return [Document(page_content="RAGLight uses dependency injection.")][:k]

    def add_documents(self, documents: list[Document]) -> None:
        pass


pipeline = RAGPipeline(
    llm=MyLLM(),
    embeddings=MyEmbeddings(),
    vector_store=MyVectorStore(),
    config=PipelineConfig(top_k=3),
)

print(pipeline.query("How is RAGLight extended?"))
```

See [`examples/v4_custom_components.py`](examples/v4_custom_components.py) for
a complete executable example.

## Built-in integrations

Import integrations explicitly so importing `raglight` never loads an optional
provider SDK:

```python
from raglight.integrations.ollama import OllamaEmbeddings, OllamaLLM
from raglight.integrations.chroma import ChromaVectorStore

embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
vector_store = ChromaVectorStore(
    collection_name="knowledge",
    embeddings_model=embeddings,
    persist_directory="./raglight-db",
)
llm = OllamaLLM(model_name="llama3.1:8b")

pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
)
```

Provider modules and public classes:

| Module | Classes |
| --- | --- |
| `raglight.integrations.openai` | `OpenAILLM`, `OpenAIEmbeddings` |
| `raglight.integrations.ollama` | `OllamaLLM`, `OllamaEmbeddings` |
| `raglight.integrations.gemini` | `GeminiLLM`, `GeminiEmbeddings` |
| `raglight.integrations.bedrock` | `BedrockLLM`, `BedrockEmbeddings` |
| `raglight.integrations.mistral` | `MistralLLM` |
| `raglight.integrations.lmstudio` | `LMStudioLLM` |
| `raglight.integrations.huggingface` | `HuggingFaceEmbeddings`, `HuggingFaceReranker` |
| `raglight.integrations.chroma` | `ChromaVectorStore` |
| `raglight.integrations.qdrant` | `QdrantVectorStore` |
| `raglight.integrations.documents` | document processors and their factory |

## Architecture

Dependencies point inward:

```text
integrations -> contracts/models <- pipeline
```

- `contracts` and `models` are dependency-free.
- `pipeline` depends only on public contracts and models.
- built-in integrations own their optional SDK dependencies.
- agentic orchestration is intentionally separate from the classic RAG API and
  will return as a dedicated v4 capability.

The detailed boundaries are documented in
[`docs/architecture.md`](docs/architecture.md), and the v3 migration table is in
[`docs/migration-v3-to-v4.md`](docs/migration-v3-to-v4.md).

## Development

The project follows test-driven development. Every behavior change starts with
a failing test, then the smallest implementation, then refactoring while the
suite remains green.

```bash
make install-dev
make check
```

`make check` runs formatting, linting, type checking, architecture rules, unit
tests, integration tests, end-to-end tests, and an isolated wheel validation.
See [`docs/testing.md`](docs/testing.md) and
[`docs/development.md`](docs/development.md) for the contribution workflow.

## License

RAGLight is distributed under the MIT License.
