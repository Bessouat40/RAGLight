<template>
  <section class="page-section">
    <h1>Examples</h1>
    <p>Examples sourced from the repository <code>examples/</code> directory.</p>
  </section>

  <section v-for="example in examples" :key="example.name" class="page-section">
    <h2>{{ example.name }}</h2>
    <p>{{ example.description }}</p>
    <CodeBlock>{{ example.code }}</CodeBlock>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import CodeBlock from '../components/CodeBlock.vue';

const examples = computed(() => [
  {
    name: 'simple_rag_api_example.py',
    description: 'RAG pipeline using the simple API.',
    code: `from raglight.rag.simple_rag_api import RAGPipeline
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
    provider=Settings.HUGGINGFACE,
    # api_base = ... # If you have a custom client URL
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RAGConfig(
        llm = Settings.DEFAULT_LLM,
        provider = Settings.OLLAMA,
        # api_base = ... # If you have a custom client URL
        # k = Settings.DEFAULT_K,
        # cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        # system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        # knowledge_base = knowledge_base
    )

pipeline = RAGPipeline(config, vector_store_config)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)`
  },
  {
    name: 'simple_agentic_rag_example.py',
    description: 'Agentic RAG pipeline with custom ignore folders.',
    code: `from raglight.config.settings import Settings
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings
from raglight.models.data_source_model import FolderSource
from raglight.models.data_source_model import GitHubSource
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

knowledge_base=[
    # FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight", branch="main"),
    ]

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
collection_name = Settings.DEFAULT_COLLECTION_NAME

vector_store_config = VectorStoreConfig(
    embedding_model = model_embeddings,
    # api_base = ... # If you have a custom client URL
    database=Settings.CHROMA,
    # host='localhost', If you want to use a remote ChromaDB
    # port='8000', If you want to use a remote ChromaDB
    persist_directory = persist_directory, # If you want to use a local ChromaDB
    provider = Settings.OLLAMA,
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
            # mcp_config=[
            #     {"url": "http://127.0.0.1:8001/sse"}
            # ],
            # api_base = ... # If you have a custom client URL
            max_steps = 5,
            api_key = Settings.MISTRAL_API_KEY, # os.environ.get('MISTRAL_API_KEY')
            ignore_folders = custom_ignore_folders,  # Use custom ignore folders
            # num_ctx = ... # Max context length
            # verbosity_level = ... # Default = 2
            knowledge_base = knowledge_base
        )

agenticRag = AgenticRAGPipeline(config, vector_store_config)

agenticRag.build()

response = agenticRag.generate("Please implement Agentic RAG for me.")

print('response : ', response)`
  },
  {
    name: 'simple_rat_api_example.py',
    description: 'RAT pipeline with reasoning LLM settings.',
    code: `from raglight.rat.simple_rat_api import RATPipeline
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
    # api_base = ... # If you have a custom client URL
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RATConfig(
        cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        # api_base = ... # If you have a custom client URL
        llm = "llama3.2:3b",
        k = Settings.DEFAULT_K,
        provider = Settings.OLLAMA,
        system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        reasoning_llm = Settings.DEFAULT_REASONING_LLM,
        reflection = 3
        # knowledge_base = knowledge_base,
    )

pipeline = RATPipeline(config, vector_store_config)

# This will ingest data from the knowledge base. Not mandatory if you have already ingested the data.
pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me the the easier python implementation")
print(response)`
  },
  {
    name: 'agentic_rag_with_mcp.py',
    description: 'Agentic RAG configured with an MCP server.',
    code: `from raglight.config.settings import Settings
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings
from raglight.models.data_source_model import FolderSource
from raglight.models.data_source_model import GitHubSource
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ]

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
collection_name = Settings.DEFAULT_COLLECTION_NAME

vector_store_config = VectorStoreConfig(
    embedding_model = model_embeddings,
    # api_base = ... # If you have a custom client URL for your embeddings provider
    database=Settings.CHROMA,
    persist_directory = persist_directory,
    provider = Settings.HUGGINGFACE,
    collection_name = collection_name
)

config = AgenticRAGConfig(
            provider = Settings.OPENAI,
            model = "gpt-4o",
            k = 10,
            system_prompt = Settings.DEFAULT_AGENT_PROMPT,
            knowledge_base = knowledge_base,
            mcp_config=[
                {"url": "http://127.0.0.1:8001/sse"}
            ],
            max_steps = 2,
            api_key = Settings.OPENAI_API_KEY, # os.environ.get('OPENAI_API_KEY')
            ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS,
            # api_base = ... # If you have a custom client URL
        )

agenticRag = AgenticRAGPipeline(config, vector_store_config)

agenticRag.build()

response = agenticRag.generate("Please implement AgenticRAGPipeline for me using RAGLight framework.")

print('response : ', response)`
  },
  {
    name: 'ingestion_example.py',
    description: 'Ingest documents into the vector store using the builder.',
    code: `from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()
Settings.setup_logging()

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
collection_name = Settings.DEFAULT_COLLECTION_NAME
data_path = os.environ.get('DATA_PATH')

vector_store = Builder() \
.with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
.with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
.build_vector_store()

vector_store.ingest(data_path=data_path)`
  },
  {
    name: 'rag_example.py',
    description: 'RAG pipeline built with the Builder helper.',
    code: `from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
model_name = 'gemma3:4b'
system_prompt_directory = Settings.DEFAULT_SYSTEM_PROMPT
collection_name = Settings.DEFAULT_COLLECTION_NAME

rag = Builder() \
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
    .with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
    .with_llm(Settings.OLLAMA, model_name=model_name, system_prompt=system_prompt_directory) \
    .build_rag(k = 5)

rag.vector_store.ingest(
                    data_path='../src/',
                    # ignore_folders=ignore_folders
                )
response = rag.generate("How can I use RAGLight to build a RAG Pipeline ?")
print(response)`
  },
  {
    name: 'discussion_example.py',
    description: 'Chat loop using an LLM built with the Builder.',
    code: `from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

model_name = 'llama3.2:3b'
system_prompt = Settings.DEFAULT_SYSTEM_PROMPT

llmOllama = Builder() \
.with_llm(Settings.OLLAMA, model_name=model_name, system_prompt=system_prompt) \
.build_llm()

# llmLMStudio = Builder() \
# .with_llm(Settings.LMStudio, model_name=model_name, system_prompt_file=system_prompt_directory) \
# .build_llm()

# llmLMStudio = Builder() \
# .with_llm(Settings.MISTRAL, model_name=model_name, system_prompt_file=system_prompt_directory) \
# .build_llm()

def chat():
    query = input(">>> ")
    if query == "quit" or query == "bye" :
        print('🤖 : See you soon 👋')
        return
    response = llmOllama.generate({"question": query})
    # response = llmLMStudio.generate({"question": query})
    print('🤖 : ', response)
    return chat()

chat()`
  },
  {
    name: 'vlm_ingestion.py',
    description: 'Vector store ingestion using a vision-language PDF processor.',
    code: `from raglight.document_processing.vlm_pdf_processor import VlmPDFProcessor
from raglight.llm.mistral_model import MistralModel
from raglight.rag.builder import Builder
from raglight.config.settings import Settings

from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

persist_directory = "./defaultDb"
model_embeddings = "nomic-embed-text:137m-v1.5-fp16"
collection_name = Settings.DEFAULT_COLLECTION_NAME
data_path = "" # Path to your data
model_name = "mistral-large-2512"

vlm = MistralModel(
    model_name=model_name,
    system_prompt="You are a technical documentation visual assistant.",
)

custom_processors = {
    "pdf": VlmPDFProcessor(vlm),           # override default PDFProcessor
}

rag = Builder() \
.with_embeddings(Settings.OLLAMA, model_name=model_embeddings) \
.with_vector_store(Settings.CHROMA,
                   persist_directory=persist_directory,
                   collection_name=collection_name,
                   custom_processors=custom_processors) \
.with_llm(Settings.MISTRAL, model_name=model_name, system_prompt="Please respond to user answer") \
.build_rag(k = 15)

rag.vector_store.ingest(data_path=data_path)

response = rag.generate("Please explain PID functionment")
print(response)`
  },
  {
    name: 'gemini_example_uses.py',
    description: 'RAG pipeline configured for Google Gemini.',
    code: `from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig

Settings.setup_logging()

knowledge_base=[
    # FolderSource(path="data/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ]

vector_store_config = VectorStoreConfig(
    embedding_model = Settings.GEMINI_EMBEDDING_MODEL,
    provider=Settings.GOOGLE_GEMINI,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RAGConfig(
        api_base = Settings.DEFAULT_GOOGLE_CLIENT,
        llm = Settings.GEMINI_LLM_MODEL,
        provider = Settings.GOOGLE_GEMINI,
        # stream = True,
        # k = Settings.DEFAULT_K,
        # cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        # system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        knowledge_base = knowledge_base
    )

pipeline = RAGPipeline(config, vector_store_config)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)`
  }
]);
</script>
