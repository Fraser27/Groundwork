import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // No rewrite: the API mounts its routers under /api, so stripping the prefix
      // made every call a 404. Override the target if the backend is not on 8010.
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8010',
        changeOrigin: true,
      },
    },
  },
})
