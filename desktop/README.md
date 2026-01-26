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
