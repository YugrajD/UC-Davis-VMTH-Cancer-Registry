import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  envDir: '..',
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
    clearMocks: true,
    coverage: {
      reporter: ['text', 'html'],
      exclude: [
        'dist/**',
        'src/test/**',
        '**/*.test.*',
        'src/main.tsx',
        'src/vite-env.d.ts',
      ],
    },
  },
});
