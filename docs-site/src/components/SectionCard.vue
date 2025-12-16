<template>
  <div class="panel card">
    <div class="card__tag">{{ tag }}</div>
    <h3>{{ title }}</h3>
    <p class="subtle">{{ description }}</p>
  </div>
</template>

<script setup>
defineProps({
  title: String,
  description: String,
  tag: String
})
</script>

<style scoped>
.card {
  padding: 22px;
  border-radius: 18px;
  display: grid;
  gap: 8px;
}

.card__tag {
  width: fit-content;
  background: rgba(56, 189, 248, 0.12);
  color: var(--text-primary);
  border: 1px solid rgba(56, 189, 248, 0.3);
  padding: 6px 10px;
  border-radius: 10px;
  font-weight: 700;
}
</style>
