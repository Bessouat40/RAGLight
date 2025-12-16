<template>
  <div class="docs-layout">
    <Sidebar />
    <section class="docs-main">
      <slot />
    </section>
  </div>
</template>

<script setup lang="ts">
import Sidebar from '../components/Sidebar.vue'
</script>

<style scoped>
.docs-layout {
  display: flex;
  gap: 1rem;
}
.docs-main {
  flex: 1;
  padding: 1rem;
}
</style>
