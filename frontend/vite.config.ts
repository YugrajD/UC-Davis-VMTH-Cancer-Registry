import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Ensure these are always defined so their %PLACEHOLDER% in index.html is
// replaced with an empty string in local dev rather than left as a literal.
if (!process.env.VITE_API_URL) {
  process.env.VITE_API_URL = '';
}
if (!process.env.VITE_COGNITO_ENDPOINT) {
  process.env.VITE_COGNITO_ENDPOINT = '';
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  envDir: '..',
  define: {
    global: 'globalThis',
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        // Server-side proxy target — use DEV_PROXY_TARGET so it can be
        // set to a docker-network hostname (e.g. http://backend:8000)
        // without leaking that hostname to the browser bundle via
        // VITE_API_URL (which client.ts reads).
        target:
          process.env.DEV_PROXY_TARGET ||
          process.env.VITE_API_URL ||
          'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
