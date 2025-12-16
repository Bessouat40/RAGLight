<template>
  <aside class="sidebar">
    <h3>Docs</h3>
    <ul>
      <li><router-link to="/docs/home/introduction">Introduction</router-link></li>
      <li><router-link to="/docs/home/installation">Installation</router-link></li>
      <li><router-link to="/docs/home/getting-started">Getting started</router-link></li>
    </ul>
  </aside>
</template>

<script setup lang="ts"></script>

<style scoped>
.sidebar {
  width: 220px;
  padding: 0.75rem;
  background: #f3f4f6;
}
.sidebar a {
  color: #111827;
  text-decoration: none;
}
</style>
