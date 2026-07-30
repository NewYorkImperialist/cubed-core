import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiPortValue = process.env.CUBED_CORE_PORT || "8000";
const apiPort = Number(apiPortValue);
if (
  !/^\d+$/.test(apiPortValue) ||
  !Number.isInteger(apiPort) ||
  apiPort < 1 ||
  apiPort > 65535
) {
  throw new Error("CUBED_CORE_PORT must be an integer from 1 to 65535");
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        // Preserve the loopback browser origin so the API can verify that the
        // automatic local-session bootstrap is same-origin even in dev.
        changeOrigin: false,
        ws: true,
      },
    },
  },
});
