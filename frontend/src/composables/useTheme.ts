import { onMounted, ref } from 'vue';

const STORAGE_KEY = 'raglight-theme';

type Theme = 'light' | 'dark';

export function useTheme() {
  const theme = ref<Theme>('light');

  const applyTheme = (value: Theme) => {
    theme.value = value;
    document.documentElement.setAttribute('data-theme', value);
    localStorage.setItem(STORAGE_KEY, value);
  };

  const toggleTheme = () => {
    applyTheme(theme.value === 'light' ? 'dark' : 'light');
  };

  onMounted(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
    if (stored === 'light' || stored === 'dark') {
      applyTheme(stored);
      return;
    }

    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(prefersDark ? 'dark' : 'light');
  });

  return { theme, toggleTheme };
}
