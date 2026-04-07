import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        // Main upload page
        main: 'index.html',
        // Results sub-page – the dynamic segment is handled at the Nginx level
        results: 'results/index.html',
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/upload':  'http://localhost:32212',
      '/status':  'http://localhost:32212',
      '/data':    'http://localhost:32212',
    },
  },
});
