<template>
  <section id="features">
    <div class="container">
      <div class="section__header">
        <h2>Features from the README</h2>
        <p class="subtle">Directly taken from the documented capabilities of RAGLight.</p>
      </div>
      <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));">
        <div v-for="feature in features" :key="feature.title" class="panel feature">
          <span class="feature__icon">{{ feature.icon }}</span>
          <h3>{{ feature.title }}</h3>
          <p class="subtle">{{ feature.description }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  features: {
    type: Array,
    required: true
  }
})
</script>

<style scoped>
h2 {
  margin: 0 0 12px 0;
  font-size: clamp(28px, 3vw, 36px);
}

.section__header {
  display: grid;
  gap: 6px;
  margin-bottom: 24px;
}

.feature {
  padding: 22px;
  border-radius: 18px;
  display: grid;
  gap: 8px;
}

.feature__icon {
  font-size: 24px;
  width: 48px;
  height: 48px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: rgba(124, 58, 237, 0.14);
}
</style>
