<template>
  <footer class="footer" id="docs">
    <div class="container footer__content">
      <div>
        <div class="nav__brand">RAGLight</div>
        <p class="subtle">Summaries taken directly from the README and examples.</p>
      </div>
      <div class="footer__links">
        <a href="https://github.com/Bessouat40/RAGLight" target="_blank" rel="noreferrer">GitHub</a>
        <a href="https://pypi.org/project/raglight/" target="_blank" rel="noreferrer">PyPI</a>
        <a href="#getting-started">Getting started</a>
        <a href="#setup">Setup</a>
        <a href="#examples">Examples</a>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.footer {
  padding: 32px 0;
  border-top: 1px solid var(--card-border);
  background: rgba(15, 23, 42, 0.9);
}

.footer__content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.footer__links {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  color: var(--text-secondary);
}

.footer__links a:hover {
  color: var(--text-primary);
}

@media (max-width: 800px) {
  .footer__content {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
