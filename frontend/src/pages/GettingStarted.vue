<template>
  <section class="page-section">
    <h1>Getting Started</h1>
    <p>Follow the steps from the README to install and run the framework.</p>
  </section>

  <section class="page-section">
    <h2>Installation</h2>
    <CodeBlock>pip install raglight</CodeBlock>
  </section>

  <section class="page-section">
    <h2>Chat with your documents</h2>
    <p>Use the CLI wizard to ingest documents and start chatting without writing code.</p>
    <CodeBlock>raglight chat</CodeBlock>
    <p>Launch the Agentic RAG wizard with:</p>
    <CodeBlock>raglight agentic-chat</CodeBlock>
  </section>

  <section class="page-section">
    <h2>Minimal RAG setup</h2>
    <p>Quickly configure a Retrieval-Augmented Generation pipeline.</p>
    <CodeBlock>
from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig

Settings.setup_logging()

knowledge_base=[
    FolderSource(path="&lt;path to your folder with pdf&gt;/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ]

vector_store_config = VectorStoreConfig(
    embedding_model = Settings.DEFAULT_EMBEDDINGS_MODEL,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RAGConfig(
        llm = Settings.DEFAULT_LLM,
        provider = Settings.OLLAMA,
    )

pipeline = RAGPipeline(config, vector_store_config)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
    </CodeBlock>
  </section>
</template>

<script setup lang="ts">
import CodeBlock from '../components/CodeBlock.vue';
</script>
