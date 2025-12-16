import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// Base set for GitHub Pages deployment. Update the base if the repository name changes.
export default defineConfig({
  plugins: [vue()],
  base: '/RAGLight/',
  build: {
    outDir: 'dist',
  },
});
