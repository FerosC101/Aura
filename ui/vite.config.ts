import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI is a control surface only: audio never passes through the browser.
// In development it talks to the Python engine on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
});
