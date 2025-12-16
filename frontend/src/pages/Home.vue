<script setup lang="ts">
import DocRenderer from '@/components/DocRenderer.vue'
</script>

<template>
  <div class="page">
    <DocRenderer path="home/introduction.md" />
    <DocRenderer path="home/installation.md" />
    <DocRenderer path="home/getting-started.md" />
  </div>
</template>
