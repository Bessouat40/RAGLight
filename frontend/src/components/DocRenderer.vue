<script setup lang="ts">
import { ref, watchEffect } from 'vue'
import { marked } from 'marked'

const props = defineProps<{
  path: string
}>()

const content = ref('')

watchEffect(async () => {
  const res = await fetch(`/docs/${props.path}`)
  content.value = await res.text()
})
</script>

<template>
  <article class="doc" v-html="marked(content)" />
</template>
