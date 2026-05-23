import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // Dev-only proxy: Replicate's REST API sends no CORS headers, so direct browser
    // calls are blocked. In dev we route them through Vite. Production (no proxy) hits
    // the documented CORS limitation — a same-origin proxy is required to deploy.
    proxy: {
      '/replicate': {
        target: 'https://api.replicate.com',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/replicate/, ''),
      },
    },
  },
});
