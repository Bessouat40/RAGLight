<template>
  <div>
    <NavBar />
    <main>
      <HeroSection />
      <FeatureGrid :features="features" />
      <ContentSection
        title="Composable building blocks"
        subtitle="Craft modular RAG pipelines that stay maintainable as your stack grows."
      >
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 24px;">
          <SectionCard
            v-for="card in architecture"
            :key="card.title"
            :title="card.title"
            :description="card.description"
            :tag="card.tag"
          />
        </div>
      </ContentSection>

      <Timeline :steps="timeline" />

      <ContentSection
        title="Batteries included documentation"
        subtitle="Get started quickly with templates for ingestion, memory, retrieval, and evaluation."
      >
        <div class="panel" style="padding: 28px; display: grid; gap: 14px;">
          <div v-for="guide in guides" :key="guide.title" class="guide">
            <div style="display: flex; align-items: center; gap: 12px;">
              <span class="badge" style="background: rgba(56,189,248,0.1); color: var(--text-primary); border-color: rgba(56,189,248,0.3);">
                {{ guide.category }}
              </span>
              <h3 style="margin: 0; font-size: 18px;">{{ guide.title }}</h3>
            </div>
            <p class="subtle" style="margin: 6px 0 0 0;">{{ guide.description }}</p>
          </div>
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
    title: 'Production grade RAG',
    description: 'Composable pipelines for ingestion, retrievers, memory, evaluation, and tracing.',
    icon: '⚡'
  },
  {
    title: 'Bring your providers',
    description: 'Use your preferred vector stores, LLMs, and orchestrators without boilerplate.',
    icon: '🔌'
  },
  {
    title: 'Observability built-in',
    description: 'Tracing hooks, metrics, and evaluation loops to keep conversations reliable.',
    icon: '📈'
  },
  {
    title: 'Python first',
    description: 'Opinionated project structure with type hints, tests, and examples ready to run.',
    icon: '🐍'
  }
]

const architecture = [
  {
    title: 'Retrieval orchestration',
    description: 'Hybrid, semantic, or keyword retrieval made pluggable with shared interfaces.',
    tag: 'Retrievers'
  },
  {
    title: 'Memory and state',
    description: 'Session memory primitives that keep interactions contextual and auditable.',
    tag: 'Memory'
  },
  {
    title: 'Data pipelines',
    description: 'Chunking, embedding, and indexing flows that you can extend with your own nodes.',
    tag: 'Ingestion'
  },
  {
    title: 'Evaluation',
    description: 'Scenarios, prompts, and scoring hooks to validate changes before you ship.',
    tag: 'Quality'
  }
]

const timeline = [
  {
    title: 'Start with the CLI',
    detail: 'Bootstrap a new project with templates for ingestion, chat, and evaluation flows.'
  },
  {
    title: 'Wire your data',
    detail: 'Connect your data sources and embed them with the providers you already trust.'
  },
  {
    title: 'Ship safely',
    detail: 'Trace interactions, benchmark outputs, and iterate with confidence using RAGLight utilities.'
  }
]

const guides = [
  {
    category: 'Quickstart',
    title: 'Install RAGLight and explore the examples',
    description: 'Spin up the CLI, ingest a sample corpus, and ship a first retrieval-augmented endpoint.'
  },
  {
    category: 'Architecture',
    title: 'Understand the building blocks',
    description: 'Learn how retrievers, memory, evaluation, and tracing fit together in a clean project layout.'
  },
  {
    category: 'Production',
    title: 'Deploy with observability',
    description: 'Use tracing hooks and evaluation suites to keep deployments stable in production.'
  }
]
</script>
