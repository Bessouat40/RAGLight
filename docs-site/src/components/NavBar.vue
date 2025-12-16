<template>
  <header class="nav">
    <div class="container nav__inner">
      <div class="nav__brand">RAGLight</div>
      <nav class="nav__links">
        <a href="#features">Features</a>
        <a href="#architecture">Architecture</a>
        <a href="#docs">Docs</a>
      </nav>
      <div class="nav__actions">
        <a class="button secondary" href="https://github.com/anthodev/raglight" target="_blank" rel="noreferrer">
          View on GitHub
        </a>
        <a class="button primary" href="#get-started">Get started</a>
      </div>
    </div>
  </header>
</template>

<style scoped>
.nav {
  position: sticky;
  top: 0;
  z-index: 20;
  backdrop-filter: blur(12px);
  background: rgba(11, 13, 23, 0.85);
  border-bottom: 1px solid var(--card-border);
}

.nav__inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 0;
  gap: 14px;
}

.nav__brand {
  font-weight: 800;
  letter-spacing: 0.5px;
}

.nav__links {
  display: flex;
  gap: 24px;
  color: var(--text-secondary);
}

.nav__links a:hover {
  color: var(--text-primary);
}

.nav__actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

@media (max-width: 900px) {
  .nav__links {
    display: none;
  }

  .nav__actions {
    width: 100%;
    justify-content: flex-end;
  }
}
</style>
