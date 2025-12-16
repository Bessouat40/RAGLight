<template>
  <section class="timeline" id="architecture">
    <div class="container">
      <div class="section__header">
        <h2>Setup flow</h2>
        <p class="subtle">Follow the same order shown in the README examples.</p>
      </div>
      <div class="timeline__steps">
        <div v-for="(step, index) in steps" :key="step.title" class="panel step">
          <div class="step__index">0{{ index + 1 }}</div>
          <div>
            <h3>{{ step.title }}</h3>
            <p class="subtle">{{ step.detail }}</p>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  steps: {
    type: Array,
    required: true
  }
})
</script>

<style scoped>
.timeline__steps {
  display: grid;
  gap: 16px;
}

.step {
  padding: 22px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 12px;
  align-items: center;
  border-radius: 18px;
}

.step__index {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: rgba(124, 58, 237, 0.16);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--accent-2);
  font-weight: 800;
}
</style>
