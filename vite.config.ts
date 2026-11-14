import { defineConfig } from 'vite';

// Relative base so the build works when served from a subpath,
// e.g. https://<user>.github.io/pixel-art-poc/ on GitHub Pages.
export default defineConfig({
  base: './',
});
