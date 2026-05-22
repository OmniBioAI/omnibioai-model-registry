import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: "/_svc/modelregistry",
  plugins: [react()],
  server: {
    proxy: {
      "/modelregistry/v1": {
        target: "http://localhost:8095",
        changeOrigin: true,
        rewrite: (p: string) => p.replace(/^\/modelregistry/, ""),
      },
    },
  },
})
