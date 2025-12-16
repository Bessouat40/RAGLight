<template>
  <aside class="sidebar">
    <div class="sidebar-section">
      <h4>Overview</h4>
      <router-link to="/">Home</router-link>
    </div>
    <div class="sidebar-section">
      <h4>Guides</h4>
      <router-link to="/getting-started">Getting Started</router-link>
      <router-link to="/docs">Documentation</router-link>
      <router-link to="/setup">Setup & Configuration</router-link>
    </div>
    <div class="sidebar-section">
      <h4>Examples</h4>
      <router-link to="/examples">Examples</router-link>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  padding: 24px;
  border-right: 1px solid var(--border-color);
  min-height: calc(100vh - 64px);
  background: var(--bg-color);
}

.sidebar-section {
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sidebar h4 {
  margin: 0;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-color);
}
</style>
