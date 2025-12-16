import { createRouter, createWebHistory } from 'vue-router'

import Home from '@/pages/Home.vue'
import Docs from '@/pages/Docs.vue'
import Examples from '@/pages/Examples.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
    },
    {
      path: '/docs',
      name: 'docs',
      component: Docs,
    },
    {
      path: '/examples',
      name: 'examples',
      component: Examples,
    },
  ],
})
