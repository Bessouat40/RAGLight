# RAGLight

**RAGLight** is a lightweight, modular Python framework for building
**Retrieval-Augmented Generation (RAG)** systems and advanced agentic pipelines.

It is designed for engineers who want **full control** over their RAG stack,
without heavyweight abstractions or hidden magic.

RAGLight focuses on **clarity, extensibility, and composability**, making it
suitable for both experimentation and production-grade systems.

---

## Why RAGLight?

Most RAG frameworks trade simplicity for abstraction.
RAGLight takes the opposite approach.

It provides:

- Explicit pipelines
- Config-driven components
- Minimal coupling between layers
- A clear mental model of how data flows

You decide:

- which LLM you use
- how documents are ingested
- how retrieval works
- how reasoning is performed

---

## Core capabilities

- 🔹 **RAG pipelines** with configurable retrieval and generation
- 🤖 **Agentic RAG** with multi-step reasoning and reflection loops
- 🧠 **RAT (Reasoning-Augmented Tasks)** using dedicated reasoning models
- 🔌 **MCP integration** for external tools (code execution, databases, search)
- 📄 **Multi-format ingestion**: PDF, code, markdown, text, repositories
- 🧩 **Pluggable architecture**: LLMs, embeddings, vector stores are interchangeable

RAGLight is **LLM-agnostic** and works with both local and remote providers.

---

## Supported providers

RAGLight currently supports:

**LLMs**

- Ollama
- LMStudio
- OpenAI
- Mistral API
- vLLM
- Google Gemini

**Embeddings**

- HuggingFace
- Ollama
- OpenAI
- vLLM
- Google Gemini

**Vector stores**

- Chroma

---

## Typical use cases

- Chat with private documents
- Codebase exploration and reasoning
- Technical documentation assistants
- Agentic systems with tool usage
- Offline or air-gapped RAG setups
- Multimodal RAG (PDFs with diagrams and images)

---

## Design philosophy

RAGLight is built around a few core principles:

- **Explicit over implicit**
- **Composition over inheritance**
- **Configuration over convention**
- **Minimal abstractions**
- **No vendor lock-in**

If you can read the code, you can understand the system.

---

## Next steps

- 👉 Go to **Installation** to set up RAGLight
- 👉 Follow **Getting Started** to build your first pipeline
- 👉 Explore **Documentation** for detailed APIs and configuration
- 👉 Check **Examples** for real-world usage patterns
