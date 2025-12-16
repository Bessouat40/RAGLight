<template>
  <pre class="code-block"><code><slot /></code></pre>
</template>

<style scoped>
.code-block {
  background: var(--code-bg);
  border: 1px solid var(--border-color);
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  white-space: pre-wrap;
}
</style>
