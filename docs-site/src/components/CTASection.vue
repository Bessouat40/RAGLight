<template>
  <section id="get-started">
    <div class="container">
      <div class="panel cta">
        <div>
          <div class="badge">Ready to build?</div>
          <h2>Start your next RAG experience with RAGLight.</h2>
          <p class="subtle">
            Clean architecture, batteries-included utilities, and documentation to help you ship faster.
          </p>
        </div>
        <div class="cta__actions">
          <a class="button primary" href="https://pypi.org/project/raglight/" target="_blank" rel="noreferrer">
            Install from PyPI
          </a>
          <a class="button secondary" href="https://github.com/anthodev/raglight" target="_blank" rel="noreferrer">
            View repository
          </a>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.cta {
  padding: 28px;
  display: grid;
  gap: 18px;
  align-items: center;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  border-radius: 20px;
}

.cta__actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

h2 {
  margin: 8px 0 0 0;
  font-size: clamp(26px, 3vw, 32px);
}
</style>
