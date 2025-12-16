<template>
  <header class="navbar">
    <div class="brand">
      <router-link to="/">RAGLight Docs</router-link>
    </div>
    <nav class="nav-links">
      <router-link to="/getting-started">Getting Started</router-link>
      <router-link to="/docs">Documentation</router-link>
      <router-link to="/setup">Setup</router-link>
      <router-link to="/examples">Examples</router-link>
      <ThemeToggle />
    </nav>
  </header>
</template>

<script setup lang="ts">
import ThemeToggle from './ThemeToggle.vue';
</script>

<style scoped>
.navbar {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-color);
}

.brand a {
  font-weight: 700;
  font-size: 18px;
}

.nav-links {
  display: flex;
  gap: 16px;
  align-items: center;
}

.nav-links a {
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid transparent;
}

.nav-links a.router-link-active {
  border-color: var(--border-color);
}
</style>
