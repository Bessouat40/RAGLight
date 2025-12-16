<template>
  <section class="hero" id="top">
    <div class="container hero__content">
      <div class="badge">Lightweight RAG library</div>
      <h1>RAGLight keeps retrieval-augmented workflows simple.</h1>
      <p class="subtle">
        Modular Python components for Retrieval-Augmented Generation: configure your providers, ingest data, and
        generate answers with clear pipelines.
      </p>
      <div class="hero__actions">
        <a class="button primary" href="#getting-started">Get started</a>
        <a class="button secondary" href="https://github.com/Bessouat40/RAGLight" target="_blank" rel="noreferrer">
          View README
        </a>
      </div>
      <div class="hero__cards">
        <div class="panel hero__card">
          <div class="hero__pill">CLI</div>
          <h3>Chat quickly</h3>
          <p class="subtle">Run <code>raglight chat</code> or <code>raglight agentic-chat</code> to test without writing code.</p>
        </div>
        <div class="panel hero__card">
          <div class="hero__pill">Pipelines</div>
          <h3>Pick your flow</h3>
          <p class="subtle">Use RAG, Agentic RAG, or RAT pipelines described in the README.</p>
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
