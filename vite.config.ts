import { defineConfig } from "vite";

// Relative base so the build works both at the dev server root and under the
// GitHub Pages project path (/pacmap-webgpu/). There is no router and every
// asset is emitted next to index.html, so relative URLs are enough.
export default defineConfig({
  base: "./",
});
