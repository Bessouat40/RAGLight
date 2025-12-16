<template>
  <div>
    <NavBar />
    <main>
      <HeroSection />

      <FeatureGrid :features="features" />

      <ContentSection
        sectionId="getting-started"
        title="Getting started"
        subtitle="Install the library, try the CLI, and build your first pipeline with the snippets from the README."
      >
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));">
          <div class="panel">
            <h3>Install</h3>
            <p class="subtle">Grab the package from PyPI.</p>
            <pre><code>pip install raglight</code></pre>
          </div>
          <div class="panel">
            <h3>Use the CLI</h3>
            <p class="subtle">Launch the chat wizards without writing code.</p>
            <pre><code>raglight chat
raglight agentic-chat</code></pre>
          </div>
          <div class="panel">
            <h3>Quick start</h3>
            <p class="subtle">Run the minimal RAG example from the README.</p>
            <pre><code>from raglight.rag.simple_rag_api import RAGPipeline
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig

vector_store_config = VectorStoreConfig(
    embedding_model=Settings.DEFAULT_EMBEDDINGS_MODEL,
    api_base=Settings.DEFAULT_OLLAMA_CLIENT,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory='./defaultDb',
    collection_name=Settings.DEFAULT_COLLECTION_NAME,
)

config = RAGConfig(
    llm=Settings.DEFAULT_LLM,
    provider=Settings.OLLAMA,
)

pipeline = RAGPipeline(config, vector_store_config)
pipeline.build()
response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ?")</code></pre>
          </div>
        </div>
      </ContentSection>

      <ContentSection
        sectionId="docs"
        title="Core elements"
        subtitle="Each block of RAGLight is described in the README so you can assemble the right pipeline."
      >
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));">
          <SectionCard
            v-for="card in coreElements"
            :key="card.title"
            :title="card.title"
            :description="card.description"
            :tag="card.tag"
          />
        </div>
      </ContentSection>

      <Timeline :steps="setupSteps" />

      <ContentSection
        sectionId="setup"
        title="Providers and setup"
        subtitle="Everything available in the README: requirements, environment variables, and provider options."
      >
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));">
          <div class="panel">
            <h3>Requirements</h3>
            <p class="subtle">Supported LLM services.</p>
            <ul class="subtle list">
              <li>Ollama</li>
              <li>Google Gemini</li>
              <li>LMStudio</li>
              <li>vLLM</li>
              <li>OpenAI API</li>
              <li>Mistral API</li>
            </ul>
          </div>
          <div class="panel">
            <h3>Environment</h3>
            <p class="subtle">Tune clients with environment variables.</p>
            <ul class="subtle list">
              <li>MISTRAL_API_KEY</li>
              <li>OLLAMA_CLIENT_URL</li>
              <li>LMSTUDIO_CLIENT</li>
              <li>OPENAI_CLIENT_URL</li>
              <li>OPENAI_API_KEY</li>
              <li>GEMINI_API_KEY</li>
            </ul>
          </div>
          <div class="panel">
            <h3>Providers</h3>
            <p class="subtle">Pick what you need for each layer.</p>
            <ul class="subtle list">
              <li>LLM: LMStudio, Ollama, Mistral, vLLM, OpenAI, Google</li>
              <li>Embeddings: HuggingFace, Ollama, vLLM, OpenAI, Google</li>
              <li>Vector store: Chroma</li>
            </ul>
          </div>
        </div>
      </ContentSection>

      <ContentSection
        sectionId="examples"
        title="Examples"
        subtitle="Direct pointers to the Python scripts in the examples folder."
      >
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));">
          <SectionCard
            v-for="example in examples"
            :key="example.title"
            :title="example.title"
            :description="example.description"
            :tag="example.tag"
          />
        </div>
      </ContentSection>

      <CTASection />
    </main>
    <FooterSection />
  </div>
</template>

<script setup>
import NavBar from './components/NavBar.vue'
import HeroSection from './components/HeroSection.vue'
import FeatureGrid from './components/FeatureGrid.vue'
import ContentSection from './components/ContentSection.vue'
import SectionCard from './components/SectionCard.vue'
import Timeline from './components/Timeline.vue'
import CTASection from './components/CTASection.vue'
import FooterSection from './components/FooterSection.vue'

const features = [
  {
    title: 'Embeddings Model Integration',
    description: 'Plug in preferred embedding models like HuggingFace all-MiniLM-L6-v2 for compact vector embeddings.',
    icon: '🧠'
  },
  {
    title: 'LLM Agnostic',
    description: 'Work with multiple providers including Ollama, LMStudio, Mistral, vLLM, OpenAI, and Google Gemini.',
    icon: '🤖'
  },
  {
    title: 'RAG Pipeline',
    description: 'Combine retrieval and generation with configurable models, prompts, and knowledge bases.',
    icon: '🔎'
  },
  {
    title: 'RAT Pipeline',
    description: 'Add reflection loops with a reasoning LLM to improve responses when needed.',
    icon: '🔁'
  },
  {
    title: 'Agentic RAG',
    description: 'Extend RAG with an agent that can query the vector store and follow multiple steps.',
    icon: '🧭'
  },
  {
    title: 'MCP Integration',
    description: 'Connect external tools via MCP servers to enrich agent reasoning.',
    icon: '🧩'
  },
  {
    title: 'Flexible Document Support',
    description: 'Ingest PDFs, text, code, and more with built-in processors and optional custom handlers.',
    icon: '📁'
  },
  {
    title: 'Extensible Architecture',
    description: 'Swap vector stores, embeddings, or LLMs with clear configuration objects.',
    icon: '🧱'
  }
]

const coreElements = [
  {
    title: 'Knowledge Base',
    description: 'Declare FolderSource or GitHubSource entries to ingest content when building your pipelines.',
    tag: 'Data'
  },
  {
    title: 'RAG Pipeline',
    description: 'Set up retrieval and generation with RAGConfig and VectorStoreConfig then call build() and generate().',
    tag: 'Retrieval'
  },
  {
    title: 'Agentic RAG Pipeline',
    description: 'Add an agent layer that can retrieve, reason over multiple steps, and respect ignore_folders.',
    tag: 'Agent'
  },
  {
    title: 'RAT Pipeline',
    description: 'Use a reasoning model and reflection steps to enhance answers beyond standard RAG.',
    tag: 'Reasoning'
  },
  {
    title: 'Custom Builder',
    description: 'Chain with_embeddings, with_vector_store, and with_llm to build custom pipelines.',
    tag: 'Builder'
  },
  {
    title: 'Override Processors',
    description: 'Register processors like VlmPDFProcessor for specific file types when ingesting.',
    tag: 'Processing'
  }
]

const setupSteps = [
  {
    title: 'Prepare your sources',
    detail: 'Collect folders or GitHub repositories you want to ingest as a knowledge base.'
  },
  {
    title: 'Configure providers',
    detail: 'Pick providers for LLM, embeddings, and vector stores; set API keys or client URLs as needed.'
  },
  {
    title: 'Build and query',
    detail: 'Call build() to ingest and use generate() to query your data with the configured pipeline.'
  }
]

const examples = [
  {
    title: 'simple_rag_api_example.py',
    description: 'Minimal RAG pipeline using RAGPipeline, VectorStoreConfig, and RAGConfig.',
    tag: 'examples/simple_rag_api_example.py'
  },
  {
    title: 'simple_agentic_rag_example.py',
    description: 'Agentic RAG pipeline with configurable steps, prompts, and ignore_folders.',
    tag: 'examples/simple_agentic_rag_example.py'
  },
  {
    title: 'simple_rat_api_example.py',
    description: 'Reasoning-augmented retrieval with RATPipeline and reflection loops.',
    tag: 'examples/simple_rat_api_example.py'
  },
  {
    title: 'agentic_rag_with_mcp.py',
    description: 'Use MCP integration to add external tools to your agent.',
    tag: 'examples/agentic_rag_with_mcp.py'
  },
  {
    title: 'ingestion_example.py',
    description: 'Ingest documents into Chroma with the builder utilities.',
    tag: 'examples/ingestion_example.py'
  },
  {
    title: 'vlm_ingestion.py',
    description: 'Register VlmPDFProcessor for vision-language PDF ingestion.',
    tag: 'examples/vlm_ingestion.py'
  }
]
</script>
