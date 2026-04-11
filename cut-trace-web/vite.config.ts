import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/cut-trace-api": {
        target: "http://localhost:7701",
        changeOrigin: true,
      },
    },
  },
});
