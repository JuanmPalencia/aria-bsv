import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/aria-bsv/',   // GitHub Pages base path
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
