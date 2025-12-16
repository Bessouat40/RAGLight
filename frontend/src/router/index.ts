import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
import Home from '../pages/Home.vue';
import GettingStarted from '../pages/GettingStarted.vue';
import Docs from '../pages/Docs.vue';
import Setup from '../pages/Setup.vue';
import Examples from '../pages/Examples.vue';

const routes: RouteRecordRaw[] = [
  { path: '/', component: Home },
  { path: '/getting-started', component: GettingStarted },
  { path: '/docs', component: Docs },
  { path: '/setup', component: Setup },
  { path: '/examples', component: Examples },
];

// Using history mode. For GitHub Pages, configure a 404.html fallback that rewrites to /index.html.
const router = createRouter({
  history: createWebHistory('/RAGLight/'),
  routes,
  scrollBehavior() {
    return { top: 0 };
  },
});

export default router;
