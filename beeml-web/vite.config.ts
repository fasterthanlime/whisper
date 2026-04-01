import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "https://souffle.dropbear-piranha.ts.net",
        secure: true,
        changeOrigin: true,
      },
    },
  },
});
