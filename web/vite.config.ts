import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The config runs in Node; declare just what it uses rather than adding Node types to the app.
declare const process: { env: Record<string, string | undefined> };

// In development the API runs separately (`uv run slm serve`); proxy /api to it.
// SLM_API points the proxy elsewhere, e.g. a test server.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.SLM_API ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
