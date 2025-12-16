<template>
  <section class="page-section">
    <h1>Documentation</h1>
  </section>

  <section class="page-section">
    <h2>What is RAGLight</h2>
    <p>RAGLight is a lightweight and modular Python library for implementing Retrieval-Augmented Generation (RAG). It enhances LLMs by combining document retrieval with natural language inference.</p>
  </section>

  <section class="page-section">
    <h2>Core concepts</h2>
    <ul>
      <li><strong>Embeddings Model Integration</strong>: Plug in embedding models for compact and efficient vector embeddings.</li>
      <li><strong>LLM Agnostic</strong>: Integrates with different LLM providers.</li>
      <li><strong>RAG, RAT, and Agentic RAG pipelines</strong> unify retrieval and generation, with options for reflection loops and agents.</li>
      <li><strong>MCP Integration</strong> adds tool capabilities via MCP servers.</li>
      <li><strong>Flexible Document Support</strong> for PDF, TXT, DOCX, Python, Javascript, and more.</li>
      <li><strong>Extensible Architecture</strong> to swap vector stores, embeddings, or LLMs.</li>
    </ul>
  </section>

  <section class="page-section">
    <h2>Knowledge</h2>
    <p>Define knowledge bases to ingest data into your vector store.</p>
    <CodeBlock>
from raglight import RAGPipeline
pipeline = RAGPipeline(knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],
    model_name="llama3",
    provider=Settings.OLLAMA,
    k=5)

pipeline.build()
    </CodeBlock>
  </section>

  <section class="page-section">
    <h2>Readers</h2>
    <p>The README focuses on retrieval pipelines; reader-specific APIs are not detailed.</p>
  </section>

  <section class="page-section">
    <h2>Embedders</h2>
    <p>Use embedding providers such as Huggingface, Ollama, vLLM, OpenAI, or Google Gemini.</p>
    <CodeBlock>
vector_store_config = VectorStoreConfig(
    embedding_model = Settings.DEFAULT_EMBEDDINGS_MODEL,
    api_base = Settings.DEFAULT_OLLAMA_CLIENT,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)
    </CodeBlock>
  </section>

  <section class="page-section">
    <h2>Vector stores</h2>
    <p>Chroma is available as the vector store option.</p>
  </section>

  <section class="page-section">
    <h2>Models</h2>
    <p>Providers include LMStudio, Ollama, Mistral API, vLLM, OpenAI, and Google Gemini.</p>
  </section>

  <section class="page-section">
    <h2>Agents</h2>
    <p>Agentic RAG adds an agent that can retrieve data from your vector store with configurable provider, model, k, max_steps, API settings, and ignore folders.</p>
    <CodeBlock>
config = AgenticRAGConfig(
            provider = Settings.MISTRAL,
            model = "mistral-large-2411",
            k = 10,
            system_prompt = Settings.DEFAULT_AGENT_PROMPT,
            max_steps = 4,
            api_key = Settings.MISTRAL_API_KEY,
            ignore_folders = custom_ignore_folders,
        )

agenticRag = AgenticRAGPipeline(config, vector_store_config)
agenticRag.build()
    </CodeBlock>
  </section>
</template>

<script setup lang="ts">
import CodeBlock from '../components/CodeBlock.vue';
</script>
