import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 120000,
  use: { baseURL: "http://127.0.0.1:3000", headless: true },
  reporter: "list",
});
