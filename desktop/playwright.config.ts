import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for JARVIS Desktop E2E tests.
 *
 * Tests run against the Vite dev server (port 1420) with mocked API responses.
 * The FastAPI backend is NOT required - all API calls are intercepted and mocked.
 */
export default defineConfig({
  // Test directory
  testDir: "./tests/e2e",

  // Run tests in parallel for speed
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry failed tests on CI
  retries: process.env.CI ? 2 : 0,

  // Limit parallel workers on CI
  workers: process.env.CI ? 1 : undefined,

  // Reporter configuration
  reporter: [
    ["html", { outputFolder: "test-results/html-report" }],
    ["list"],
  ],

  // Shared settings for all projects
  use: {
    // Base URL for navigation
    baseURL: "http://localhost:1420",

    // Collect trace on first retry
    trace: "on-first-retry",

    // Screenshot on failure
    screenshot: "only-on-failure",

    // Video on failure
    video: "retain-on-failure",
  },

  // Test timeout
  timeout: 30000,

  // Expect timeout for assertions
  expect: {
    timeout: 5000,
  },

  // Configure projects for different browsers
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
  ],

  // Start the Vite dev server before running tests
  webServer: {
    command: "npm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },

  // Output directory for test artifacts
  outputDir: "test-results/artifacts",
});
