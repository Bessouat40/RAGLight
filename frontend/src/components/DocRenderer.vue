<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { marked } from 'marked'

const props = defineProps<{
  path: string
}>()

const html = ref('')
const isLoading = ref(false)
const error = ref('')

const renderMarkdown = (markdown: string) => {
  marked.setOptions({
    gfm: true,
    breaks: true,
  })
  html.value = marked.parse(markdown)
}

const loadDoc = async () => {
  if (!props.path) return
  isLoading.value = true
  error.value = ''
  html.value = ''

  try {
    const response = await fetch(`/docs/${props.path}`)

    if (!response.ok) {
      throw new Error(`Unable to load ${props.path}`)
    }

    const markdown = await response.text()
    renderMarkdown(markdown)
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load content'
  } finally {
    isLoading.value = false
  }
}

watch(
  () => props.path,
  () => {
    loadDoc()
  },
  { immediate: true }
)

onMounted(() => {
  if (!html.value && !isLoading.value && !error.value) {
    loadDoc()
  }
})
</script>

<template>
  <section class="doc-renderer">
    <div v-if="isLoading" class="doc-state">Loading documentation…</div>
    <div v-else-if="error" class="doc-state error">{{ error }}</div>
    <article v-else class="doc-body" v-html="html" />
  </section>
</template>

<style scoped>
.doc-renderer {
  position: relative;
  margin: 56px 0;
  padding-bottom: 8px;
}

.doc-state {
  padding: 18px 20px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.02);
  color: var(--text-muted);
  border-radius: 12px;
  font-size: 14px;
  letter-spacing: 0.01em;
}

.doc-state.error {
  color: #ffb4a2;
  border-color: rgba(255, 122, 122, 0.35);
  background: rgba(255, 122, 122, 0.05);
}

.doc-body {
  width: 100%;
}
</style>
