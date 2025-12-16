<template>
  <nav class="navbar">
    <div class="logo">RAGLight</div>

    <div class="links">
      <RouterLink to="/">Home</RouterLink>
      <RouterLink to="/docs">Documentation</RouterLink>
      <RouterLink to="/examples">Examples</RouterLink>
    </div>
  </nav>
</template>

<style scoped>
.navbar {
  position: sticky;
  top: 0;
  z-index: 10;

  display: flex;
  justify-content: space-between;
  align-items: center;

  padding: 14px 32px;
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border);
}

.logo {
  font-weight: 600;
  letter-spacing: 0.3px;
  color: white;
}

.links a {
  margin-left: 28px;
  font-size: 14px;
  color: var(--text-soft);
  text-decoration: none;
  position: relative;
}

.links a:hover {
  color: white;
}

.links a.router-link-active {
  color: var(--accent);
}

.links a.router-link-active::after {
  content: '';
  position: absolute;
  bottom: -6px;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--accent);
  border-radius: 2px;
}
</style>
