[![MSeeP.ai Security Assessment Badge](https://mseep.net/pr/bessouat40-raglight-badge.png)](https://mseep.ai/app/bessouat40-raglight)

# RAGLight

![License](https://img.shields.io/github/license/Bessouat40/RAGLight)
[![Downloads](https://static.pepy.tech/personalized-badge/raglight?period=total&units=international_system&left_color=grey&right_color=red&left_text=Downloads)](https://pepy.tech/projects/raglight)

<div align="center">
    <img alt="RAGLight" height="200px" src="./media/raglight.png">
</div>

**RAGLight** is a lightweight and modular Python library for implementing **Retrieval-Augmented Generation (RAG)**. It enhances the capabilities of Large Language Models (LLMs) by combining document retrieval with natural language inference.

Designed for simplicity and flexibility, RAGLight provides modular components to easily integrate various LLMs, embeddings, and vector stores, making it an ideal tool for building context-aware AI solutions.

---

## 📚 Table of Contents

- [Requirements](#⚠️-requirements)
- [Features](#features)
- [Import library](#import-library-🛠️)
- [Chat with Your Documents Instantly With CLI](#chat-with-your-documents-instantly-with-cli-💬)

  - [Ignore Folders Feature](#ignore-folders-feature-🚫)
  - [Ignore Folders in Configuration Classes](#ignore-folders-in-configuration-classes-🚫)

- [Environment Variables](#environment-variables)
- [Providers and Databases](#providers-and-databases)

  - [LLM](#llm)
  - [Embeddings](#embeddings)
  - [Vector Store](#vector-store)

- [Quick Start](#quick-start-🚀)

  - [Knowledge Base](#knowledge-base)
  - [RAG](#rag)
  - [Agentic RAG](#agentic-rag)
  - [MCP Integration](#mcp-integration)
  - [RAT](#rat)
  - [Use Custom Pipeline](#use-custom-pipeline)

- [Use RAGLight with Docker](#use-raglight-with-docker)

  - [Build your image](#build-your-image)
  - [Run your image](#run-your-image)

---

> ## ⚠️ Requirements
>
> Actually RAGLight supports :
>
> - Ollama
> - Google
> - LMStudio
> - vLLM
> - OpenAI API
> - Mistral API
>
> If you use LMStudio, you need to have the model you want to use loaded in LMStudio.

## Features

- **Embeddings Model Integration**: Plug in your preferred embedding models (e.g., HuggingFace **all-MiniLM-L6-v2**) for compact and efficient vector embeddings.
- **LLM Agnostic**: Seamlessly integrates with different LLMs from different providers (Ollama and LMStudio supported).
- **RAG Pipeline**: Combines document retrieval and language generation in a unified workflow.
- **RAT Pipeline**: Combines document retrieval and language generation in a unified workflow. Add reflection loops using a reasoning model like **Deepseek-R1** or **o1**.
- **Agentic RAG Pipeline**: Use Agent to improve your RAG performances.
- 🔌 **MCP Integration**: Add external tool capabilities (e.g. code execution, database access) via MCP servers.
- **Flexible Document Support**: Ingest and index various document types (e.g., PDF, TXT, DOCX, Python, Javascript, ...).
- **Extensible Architecture**: Easily swap vector stores, embedding models, or LLMs to suit your needs.

---

## Import library 🛠️

To install the library, run:

```bash
pip install raglight
```

---

## Chat with Your Documents Instantly With CLI 💬

For the quickest and easiest way to get started, RAGLight provides an interactive command-line wizard. It will guide you through every step, from selecting your documents to chatting with them, without writing a single line of Python.
Prerequisite: Ensure you have a local LLM service like Ollama running.

Just run this one command in your terminal:

```bash
raglight chat
```

You can also launch the Agentic RAG wizard with:

```bash
raglight agentic-chat
```

The wizard will guide you through the setup process. Here is what it looks like:

<div align="center">
    <img alt="RAGLight" src="./media/cli.png">
</div>

The wizard will ask you for:

- 📂 Data Source: The path to your local folder containing the documents.
- 🚫 Ignore Folders: Configure which folders to exclude during indexing (e.g., `.venv`, `node_modules`, `__pycache__`).
- 💾 Vector Database: Where to store the indexed data and what to name it.
- 🧠 Embeddings Model: Which model to use for understanding your documents.
- 🤖 Language Model (LLM): Which LLM to use for generating answers.

After configuration, it will automatically index your documents and start a chat session.

### Ignore Folders Feature 🚫

RAGLight automatically excludes common directories that shouldn't be indexed, such as:

- Virtual environments (`.venv`, `venv`, `env`)
- Node.js dependencies (`node_modules`)
- Python cache files (`__pycache__`)
- Build artifacts (`build`, `dist`, `target`)
- IDE files (`.vscode`, `.idea`)
- And many more...

You can customize this list during the CLI setup or use the default configuration. This ensures that only relevant code and documentation are indexed, improving performance and reducing noise in your search results.

### Ignore Folders in Configuration Classes 🚫

The ignore folders feature is also available in all configuration classes, allowing you to specify which directories to exclude during indexing:

- **RAGConfig**: Use `ignore_folders` parameter to exclude folders during RAG pipeline indexing
- **AgenticRAGConfig**: Use `ignore_folders` parameter to exclude folders during AgenticRAG pipeline indexing
- **RATConfig**: Use `ignore_folders` parameter to exclude folders during RAT pipeline indexing
- **VectorStoreConfig**: Use `ignore_folders` parameter to exclude folders during vector store operations

All configuration classes use `Settings.DEFAULT_IGNORE_FOLDERS` as the default value, but you can override this with your custom list:

```python
# Example: Custom ignore folders for any configuration
custom_ignore_folders = [
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".git",
    "build",
    "dist",
    "temp_files",  # Your custom folders
    "cache"
]

# Use in any configuration class
config = RAGConfig(
    llm=Settings.DEFAULT_LLM,
    provider=Settings.OLLAMA,
    ignore_folders=custom_ignore_folders  # Override default
)
```

See the complete example in [examples/ignore_folders_config_example.py](examples/ignore_folders_config_example.py) for all configuration types.

## Environment Variables

You can set several environment vaiables to change **RAGLight** settings :

- `MISTRAL_API_KEY` if you want to use Mistral API
- `OLLAMA_CLIENT_URL` if you have a custom Ollama URL
- `LMSTUDIO_CLIENT` if you have a custom LMStudio URL
- `OPENAI_CLIENT_URL` if you have a custom OpenAI URL or vLLM URL
- `OPENAI_API_KEY` if you need an OpenAI key
- `GEMINI_API_KEY` if you need an OpenAI key

## Providers and databases

### LLM

For your LLM inference, you can use these providers :

- LMStudio (`Settings.LMSTUDIO`)
- Ollama (`Settings.OLLAMA`)
- Mistral API (`Settings.MISTRAL`)
- vLLM (`Settings.VLLM`)
- OpenAI (`Settings.OPENAI`)
- Google (`Settings.GOOGLE_GEMINI`)

### Embeddings

For embeddings models, you can use these providers :

- Huggingface (`Settings.HUGGINGFACE`)
- Ollama (`Settings.OLLAMA`)
- vLLM (`Settings.VLLM`)
- OpenAI (`Settings.OPENAI`)
- Google (`Settings.GOOGLE_GEMINI`)

### Vector Store

For your vector store, you can use :

- Chroma (`Settings.CHROMA`)

## Quick Start 🚀

### Knowledge Base

Knowledge Base is a way to define data you want to ingest inside your vector store during the initialization of your RAG.
It's the data ingest when you call `build` function :

```python
from raglight import RAGPipeline
pipeline = RAGPipeline(knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],
    model_name="llama3",
    provider=Settings.OLLAMA,
    k=5)

pipeline.build()
```

You can define two different knowledge base :

1. Folder Knowledge Base

All files/folders into this directory will be ingested inside the vector store :

```python
from raglight import FolderSource
FolderSource(path="<path to your folder with pdf>/knowledge_base"),
```

2. Github Knowledge Base

You can declare Github Repositories you want to store into your vector store :

```python
from raglight import GitHubSource
GitHubSource(url="https://github.com/Bessouat40/RAGLight")
```

### RAG

You can setup easily your RAG with RAGLight :

```python
from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig

Settings.setup_logging()

knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ]

vector_store_config = VectorStoreConfig(
    embedding_model = Settings.DEFAULT_EMBEDDINGS_MODEL,
    api_base = Settings.DEFAULT_OLLAMA_CLIENT,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RAGConfig(
        llm = Settings.DEFAULT_LLM,
        provider = Settings.OLLAMA,
        # k = Settings.DEFAULT_K,
        # cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        # system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        # knowledge_base = knowledge_base
    )

pipeline = RAGPipeline(config, vector_store_config)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
```

You just have to fill the model you want to use.

> ⚠️
> By default, LLM Provider will be Ollama

### Agentic RAG

This pipeline extends the Retrieval-Augmented Generation (RAG) concept by incorporating
an additional Agent. This agent can retrieve data from your vector store.

You can modify several parameters in your config :

- `provider` : Your LLM Provider (Ollama, LMStudio, Mistral)
- `model` : The model you want to use
- `k` : The number of document you'll retrieve
- `max_steps` : Max reflexion steps used by your Agent
- `api_key` : Your Mistral API key
- `api_base` : Your API URL (Ollama URL, LM Studio URL, ...)
- `num_ctx` : Your context max_length
- `verbosity_level` : Your logs' verbosity level
- `ignore_folders` : List of folders to exclude during indexing (e.g., [".venv", "node_modules", "__pycache__"])

```python
from raglight.config.settings import Settings
from raglight.rag.agentic_rag import AgenticRAG
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
collection_name = Settings.DEFAULT_COLLECTION_NAME

vector_store_config = VectorStoreConfig(
    embedding_model = model_embeddings,
    api_base = Settings.DEFAULT_OLLAMA_CLIENT,
    database=Settings.CHROMA,
    persist_directory = persist_directory,
    provider = Settings.HUGGINGFACE,
    collection_name = collection_name
)

# Custom ignore folders - you can override the default list
custom_ignore_folders = [
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".git",
    "build",
    "dist",
    "my_custom_folder_to_ignore"  # Add your custom folders here
]

config = AgenticRAGConfig(
            provider = Settings.MISTRAL,
            model = "mistral-large-2411",
            k = 10,
            system_prompt = Settings.DEFAULT_AGENT_PROMPT,
            max_steps = 4,
            api_key = Settings.MISTRAL_API_KEY, # os.environ.get('MISTRAL_API_KEY')
            ignore_folders = custom_ignore_folders,  # Use custom ignore folders
            # api_base = ... # If you have a custom client URL
            # num_ctx = ... # Max context length
            # verbosity_level = ... # Default = 2
            # knowledge_base = knowledge_base
        )

agenticRag = AgenticRAG(config, vector_store_config)

response = agenticRag.generate("Please implement for me AgenticRAGPipeline inspired by RAGPipeline and AgenticRAG and RAG")

print('response : ', response)
```

### MCP Integration

RAGLight supports MCP Server integration to enhance the reasoning capabilities of your agent. MCP allows the agent to interact with external tools (e.g., code execution environments, database tools, or search agents) via a standardized server interface.

To use MCP, simply pass a mcp_config parameter to your AgenticRAGConfig, where each config defines the url (and optionally transport) of the MCP server.

Just add this parameter to your AgenticRAGPipeline :

```python
config = AgenticRAGConfig(
    provider = Settings.OPENAI,
    model = "gpt-4o",
    k = 10,
    mcp_config = [
        {"url": "http://127.0.0.1:8001/sse"}  # Your MCP server URL
    ],
    ...
)
```

> 📚 Documentation: Learn how to configure and launch an MCP server using [MCPClient.server_parameters](https://huggingface.co/docs/smolagents/en/reference/tools#smolagents.MCPClient.server_parameters)

### RAT

This pipeline extends the Retrieval-Augmented Generation (RAG) concept by incorporating
an additional reasoning step using a specialized reasoning language model (LLM).

```python
from raglight.rat.simple_rat_api import RATPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rat_config import RATConfig
from raglight.config.vector_store_config import VectorStoreConfig

Settings.setup_logging()

knowledge_base=[
    FolderSource(path="<path to the folder you want to ingest into your knowledge base>"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ]

vector_store_config = VectorStoreConfig(
    embedding_model = Settings.DEFAULT_EMBEDDINGS_MODEL,
    api_base = Settings.DEFAULT_OLLAMA_CLIENT,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RATConfig(
        cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        llm = "llama3.2:3b",
        k = Settings.DEFAULT_K,
        provider = Settings.OLLAMA,
        system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        reasoning_llm = Settings.DEFAULT_REASONING_LLM,
        reflection = 3
        # knowledge_base = knowledge_base,
    )

pipeline = RATPipeline(config)

# This will ingest data from the knowledge base. Not mandatory if you have already ingested the data.
pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me the the easier python implementation")
print(response)
```

### Use Custom Pipeline

**1. Configure Your Pipeline**

You can also setup your own Pipeline :

```python
from raglight.rag.builder import Builder
from raglight.config.settings import Settings

rag = Builder() \
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
    .with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
    .with_llm(Settings.OLLAMA, model_name=model_name, system_prompt_file=system_prompt_directory, provider=Settings.LMStudio) \
    .build_rag(k = 5)
```

**2. Ingest Documents Inside Your Vector Store**

Then you can ingest data into your vector store.

1. You can use default pipeline that'll ingest no code data :

```python
rag.vector_store.ingest(data_path='./data')
```

2. Or you can use code pipeline :

```python
rag.vector_store.ingest(repos_path=['./repository1', './repository2'])
```

This pipeline will ingest code embeddings into your collection : **collection_name**.
But this pipeline will also extract all signatures from your code base and ingest it into : **collection_name_classes**.

You have access to two different functions inside `VectorStore` class : `similarity_search` and `similarity_search_class` to search into different collection.

**3. Query the Pipeline**

Retrieve and generate answers using the RAG pipeline:

```python
response = rag.generate("How can I optimize my marathon training?")
print(response)
```

> ### ✚ More Examples
>
> You can find more examples for all these use cases in the [examples](https://github.com/Bessouat40/RAGLight/blob/main/examples) directory.

## Use RAGLight with Docker

You can use RAGLight inside a Docker container easily.
Find Dockerfile example here : [examples/Dockerfile.example](https://github.com/Bessouat40/RAGLight/blob/main/examples/Dockerfile.example)

### Build your image

Just go to **examples** directory and run :

```bash
docker build -t docker-raglight -f Dockerfile.example .
```

## Run your image

In order your container can communicate with Ollama or LMStudio, you need to add a custom host-to-IP mapping :

```bash
docker run --add-host=host.docker.internal:host-gateway docker-raglight
```

We use `--add-host` flag to allow Ollama call.
