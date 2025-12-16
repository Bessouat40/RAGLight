<template>
  <section :id="sectionId">
    <div class="container">
      <div class="section__header">
        <h2>{{ title }}</h2>
        <p class="subtle">{{ subtitle }}</p>
      </div>
      <div>
        <slot />
      </div>
    </div>
  </section>
</template>

<script setup>
const props = defineProps({
  title: {
    type: String,
    required: true
  },
  subtitle: {
    type: String,
    required: true
  },
  sectionId: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
h2 {
  margin: 0 0 8px 0;
  font-size: clamp(28px, 3vw, 36px);
}

.section__header {
  display: grid;
  gap: 6px;
  margin-bottom: 18px;
}
</style>
