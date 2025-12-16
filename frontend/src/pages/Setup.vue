<template>
  <section class="page-section">
    <h1>Setup & Configuration</h1>
    <p>Options available directly from the README.</p>
  </section>

  <section class="page-section">
    <h2>Vector store</h2>
    <ul>
      <li>Chroma (Settings.CHROMA)</li>
    </ul>
  </section>

  <section class="page-section">
    <h2>Embedding providers</h2>
    <ul>
      <li>Huggingface (Settings.HUGGINGFACE)</li>
      <li>Ollama (Settings.OLLAMA)</li>
      <li>vLLM (Settings.VLLM)</li>
      <li>OpenAI (Settings.OPENAI)</li>
      <li>Google Gemini (Settings.GOOGLE_GEMINI)</li>
    </ul>
  </section>

  <section class="page-section">
    <h2>LLM providers</h2>
    <ul>
      <li>LMStudio (Settings.LMSTUDIO)</li>
      <li>Ollama (Settings.OLLAMA)</li>
      <li>Mistral API (Settings.MISTRAL)</li>
      <li>vLLM (Settings.VLLM)</li>
      <li>OpenAI (Settings.OPENAI)</li>
      <li>Google Gemini (Settings.GOOGLE_GEMINI)</li>
    </ul>
  </section>

  <section class="page-section">
    <h2>Environment variables</h2>
    <ul>
      <li>MISTRAL_API_KEY</li>
      <li>OLLAMA_CLIENT_URL</li>
      <li>LMSTUDIO_CLIENT</li>
      <li>OPENAI_CLIENT_URL</li>
      <li>OPENAI_API_KEY</li>
      <li>GEMINI_API_KEY</li>
    </ul>
  </section>

  <section class="page-section">
    <h2>Configuration examples</h2>
    <h3>Vector store configuration</h3>
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

    <h3>RAG pipeline</h3>
    <CodeBlock>
config = RAGConfig(
        llm = Settings.DEFAULT_LLM,
        provider = Settings.OLLAMA,
    )

pipeline = RAGPipeline(config, vector_store_config)
pipeline.build()
    </CodeBlock>

    <h3>Agentic RAG pipeline</h3>
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

    <h3>RAT pipeline</h3>
    <CodeBlock>
config = RATConfig(
        cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        llm = "llama3.2:3b",
        k = Settings.DEFAULT_K,
        provider = Settings.OLLAMA,
        system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        reasoning_llm = Settings.DEFAULT_REASONING_LLM,
        reflection = 3
    )

pipeline = RATPipeline(config)
pipeline.build()
    </CodeBlock>
  </section>
</template>

<script setup lang="ts">
import CodeBlock from '../components/CodeBlock.vue';
</script>
