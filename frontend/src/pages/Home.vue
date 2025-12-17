<script setup lang="ts">
import DocRenderer from '@/components/DocRenderer.vue'
</script>

<template>
  <main class="page doc-page">
    <header class="page-hero">
      <p class="eyebrow">RAGLight Documentation</p>
      <h1>Ship retrieval-augmented systems with confidence.</h1>
      <p class="lede">
        Concise guides for building, deploying, and maintaining production-ready workflows
        with the RAGLight Python framework.
      </p>
    </header>

    <div class="doc-sections">
      <DocRenderer path="home/introduction.md" />
      <DocRenderer path="home/installation.md" />
      <DocRenderer path="home/getting-started.md" />
    </div>
  </main>
</template>

<style scoped>
.page-hero {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 36px;
}

.eyebrow {
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0;
}

h1 {
  font-size: clamp(2rem, 3vw + 1rem, 2.8rem);
  line-height: 1.2;
  margin: 0;
}

.lede {
  color: var(--text-soft);
  font-size: 1.05rem;
  max-width: 820px;
  margin: 0;
}

.doc-sections {
  display: flex;
  flex-direction: column;
}
</style>
