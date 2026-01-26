# JARVIS Desktop App

A Tauri + Svelte desktop application for JARVIS, providing a menu bar + window hybrid interface for viewing iMessage conversations.

## Prerequisites

Before you can build the desktop app, install these dependencies:

1. **Rust** (for Tauri backend):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Node.js 18+** (for Svelte frontend):
   ```bash
   brew install node
   ```

3. **Xcode Command Line Tools** (for macOS builds):
   ```bash
   xcode-select --install
   ```

## Quick Start

### 1. Start the Python API

In one terminal:
```bash
# From the project root
make api-dev
```

This starts the FastAPI server on `http://localhost:8742`.

### 2. Start the Desktop App (Development)

In another terminal:
```bash
# Install npm dependencies (first time only)
make desktop-setup

# Start Tauri dev mode
cd desktop && npm run tauri dev
```

## Building for Production

```bash
make desktop-build
```

The built app will be in `desktop/src-tauri/target/release/bundle/`.

## Generating App Icons

Before building, you need to generate the required icon files:

1. Create a 1024x1024 PNG source icon
2. Generate all required formats:
   ```bash
   cd desktop
   npm run tauri icon path/to/your/icon.png
   ```

## Architecture

```
Desktop App (Tauri)     →  Python API (FastAPI)  →  JARVIS Backend
  - Svelte UI                - /conversations        - iMessage Reader
  - Menu bar icon            - /health               - Memory Controller
  - Window management        - Port 8742             - MLX Model
```

## Features

- **Menu Bar Icon**: Left-click toggles window, right-click shows menu
- **Dashboard**: Overview of conversations and system stats
- **Messages**: Browse conversations and view messages
- **Health**: System status, memory usage, permissions

## Troubleshooting

### "Permission denied" for iMessages

Grant Full Disk Access:
1. Open System Settings → Privacy & Security → Full Disk Access
2. Add your terminal application (Terminal.app, iTerm, etc.)
3. Restart the API server

### API connection failed

Ensure the Python API is running:
```bash
curl http://localhost:8742/health
```

### Tauri build fails

Make sure you have:
- Rust installed (`rustc --version`)
- Node.js 18+ (`node --version`)
- Xcode CLI tools (`xcode-select -p`)

## Testing

The desktop app includes comprehensive E2E tests using Playwright. Tests run against the Vite dev server with mocked API responses, so no backend is required.

### Prerequisites

Install Playwright browsers (first time only):
```bash
npx playwright install
```

### Running Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run tests with UI mode (interactive debugging)
npm run test:e2e:ui

# Run tests in headed mode (see browser)
npm run test:e2e:headed

# Run tests with step-by-step debugging
npm run test:e2e:debug
```

### Test Structure

```
tests/
├── e2e/                          # E2E test files
│   ├── fixtures.ts               # Shared test fixtures
│   ├── test_app_launch.spec.ts   # App startup tests
│   ├── test_conversation_list.spec.ts
│   ├── test_message_view.spec.ts
│   ├── test_health_status.spec.ts
│   ├── test_settings.spec.ts
│   ├── test_search.spec.ts
│   └── test_ai_draft.spec.ts
└── mocks/                        # API mock data and handlers
    ├── api-data.ts               # Mock response data
    ├── api-handlers.ts           # Request interceptors
    └── index.ts                  # Mock exports
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_app_launch.spec.ts` | App opens without errors, sidebar navigation, connection status |
| `test_conversation_list.spec.ts` | Conversations load and display, selection, group indicators |
| `test_message_view.spec.ts` | Messages display, sent/received styling, timestamps |
| `test_health_status.spec.ts` | Health metrics, status banner, refresh functionality |
| `test_settings.spec.ts` | Settings load/save, model selection, sliders |
| `test_search.spec.ts` | Search input, filtering behavior |
| `test_ai_draft.spec.ts` | Draft panel, suggestion generation, selection |

### API Mocking

Tests mock all API calls to `localhost:8742` using Playwright's route interception. This means:

- **No backend required** - Tests run entirely against mock data
- **Deterministic results** - Same mock data every time
- **Error scenario testing** - Can simulate API failures

Mock data is defined in `tests/mocks/api-data.ts` and matches the types from `src/lib/api/types.ts`.

### Writing New Tests

1. Import fixtures from `./fixtures.ts`:
   ```typescript
   import { test, expect, waitForAppLoad } from "./fixtures";
   ```

2. Use `mockedPage` fixture for standard tests:
   ```typescript
   test("my test", async ({ mockedPage: page }) => {
     await page.goto("/");
     await waitForAppLoad(page);
     // ... test code
   });
   ```

3. Use `errorPage` fixture to test error handling:
   ```typescript
   test("handles errors", async ({ errorPage: page }) => {
     await page.goto("/");
     // API calls will return errors
   });
   ```

4. Use `disconnectedPage` fixture for offline scenarios:
   ```typescript
   test("handles disconnection", async ({ disconnectedPage: page }) => {
     await page.goto("/");
     // All API calls will fail with connection refused
   });
   ```

### Test Artifacts

Test results are stored in:
- `test-results/html-report/` - HTML test report
- `test-results/artifacts/` - Screenshots, videos on failure

View the HTML report:
```bash
npx playwright show-report test-results/html-report
```
