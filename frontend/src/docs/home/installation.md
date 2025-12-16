# Installation

RAGLight is distributed as a Python package and can be installed via `pip`.

It works with both **local LLM runtimes** and **remote APIs**, depending on your setup.

---

## Requirements

- Python **3.9+**
- A vector store backend (Chroma supported)
- At least one LLM provider (local or remote)

RAGLight does not ship models itself.
You must have access to an LLM runtime or API.

---

## Install via pip

```bash
pip install raglight
```

This installs the core library and all required dependencies.

---

## LLM providers

RAGLight is **LLM-agnostic**.
You can use either local runtimes or hosted APIs.

### Local providers (recommended for development)

- **Ollama**
- **LMStudio**
- **vLLM**

Make sure the model you want to use is already running.

Example with Ollama:

```bash
ollama run llama3
```

---

### Remote providers

- OpenAI
- Mistral API
- Google Gemini

For these providers, you must configure API keys.

---

## Environment variables

Depending on your provider, define the following variables:

```bash
# OpenAI / vLLM
export OPENAI_API_KEY=your_key
export OPENAI_CLIENT_URL=https://api.openai.com/v1

# Mistral
export MISTRAL_API_KEY=your_key

# Ollama (optional)
export OLLAMA_CLIENT_URL=http://localhost:11434

# LMStudio (optional)
export LMSTUDIO_CLIENT=http://localhost:1234
```

You can also place them in a `.env` file.

---

## Verify installation

You can verify that RAGLight is correctly installed by importing it:

```python
import raglight
print(raglight.__version__)
```

If no error is raised, you are ready to continue.

---

## CLI (optional)

RAGLight ships with a built-in CLI to quickly chat with your documents:

```bash
raglight chat
```

This will launch an interactive wizard that guides you through
document selection, indexing, and querying.

---

➡️ Next: continue with **Getting Started** to build your first pipeline.
