<template>
  <button class="toggle" type="button" @click="toggleTheme">
    {{ theme === 'light' ? 'Dark mode' : 'Light mode' }}
  </button>
</template>

<script setup lang="ts">
import { useTheme } from '../composables/useTheme';

const { theme, toggleTheme } = useTheme();
</script>

<style scoped>
.toggle {
  border: 1px solid var(--border-color);
  background: var(--bg-color);
  color: var(--text-color);
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
}
</style>
