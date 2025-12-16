<template>
  <section class="hero" id="top">
    <div class="container hero__content">
      <div class="badge">Modern RAG framework</div>
      <h1>Build reliable retrieval augmented apps faster with RAGLight.</h1>
      <p class="subtle">
        A concise framework for shipping production-grade RAG: ingestion, retrievers, memory, evaluation, and
        observability — ready to compose with your stack.
      </p>
      <div class="hero__actions">
        <a class="button primary" href="#get-started">Get started</a>
        <a class="button secondary" href="https://github.com/anthodev/raglight" target="_blank" rel="noreferrer">
          Explore on GitHub
        </a>
      </div>
      <div class="hero__cards">
        <div class="panel hero__card">
          <div class="hero__pill">CLI</div>
          <h3>Spin up a RAG workspace</h3>
          <p class="subtle">Use the CLI to scaffold ingestion, chat, and evaluation flows in minutes.</p>
        </div>
        <div class="panel hero__card">
          <div class="hero__pill">Observability</div>
          <h3>Trace every interaction</h3>
          <p class="subtle">Built-in hooks keep your sessions debuggable with metrics and evaluation reports.</p>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.hero__content {
  display: grid;
  gap: 20px;
  text-align: left;
}

h1 {
  margin: 0;
  font-size: clamp(32px, 5vw, 54px);
}

.hero__actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.hero__cards {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.hero__card {
  padding: 24px;
  border-radius: 20px;
}

.hero__pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(124, 58, 237, 0.16);
  color: var(--accent-2);
  font-weight: 700;
  border: 1px solid rgba(124, 58, 237, 0.3);
}
</style>
