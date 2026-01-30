# JARVIS v2

A local-first AI assistant for macOS that provides intelligent iMessage reply suggestions using MLX-based language models. Runs entirely on Apple Silicon with no cloud data transmission.

## Features

- **Smart Reply Suggestions**: Context-aware reply generation based on conversation history
- **Style Learning**: Learns your texting style from past messages
- **Semantic Search**: Find messages by meaning, not just keywords
- **Contact Profiles**: Rich analysis of communication patterns per contact
- **Template Fast-Path**: Instant responses for common reply patterns
- **Desktop App**: Native macOS app via Tauri + Svelte

## Requirements

- macOS 11+ (Big Sur or later)
- Apple Silicon (M1/M2/M3) - required for MLX
- 8GB+ RAM (16GB recommended)
- Full Disk Access permission for iMessage
- Python 3.11+
- Node.js 18+ and pnpm (for desktop app)

## Quick Start

```bash
# 1. Install Python dependencies (from project root)
cd /path/to/jarvis-ai-assistant
uv sync

# 2. Install desktop dependencies
cd v2
make install

# 3. Start the app
make app
```

The desktop app will start with:
- API server on `http://localhost:8000`
- Native Tauri window

## Project Structure

```
v2/
├── api/                 # FastAPI REST layer
│   ├── main.py         # App setup and routing
│   ├── schemas.py      # Pydantic models
│   └── routes/         # Endpoint implementations
│
├── core/               # Core business logic
│   ├── models/         # MLX model loading
│   ├── generation/     # Reply generation pipeline
│   ├── imessage/       # iMessage database access
│   ├── embeddings/     # Vector search & profiles
│   └── templates/      # Response template matching
│
├── desktop/            # Tauri/Svelte frontend
│   ├── src/            # Svelte components
│   └── src-tauri/      # Tauri configuration
│
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Documentation

- [Architecture](ARCHITECTURE.md) - System design and patterns
- [API Reference](API_REFERENCE.md) - REST API documentation
- [Generation Pipeline](GENERATION_PIPELINE.md) - How reply generation works
- [Development Guide](DEVELOPMENT.md) - Setup and workflow
- [Embeddings & Search](EMBEDDINGS.md) - Vector search system

## Available Commands

```bash
make help      # Show all commands
make app       # Start API + Desktop app
make dev       # Start API + Browser dev mode
make api       # Start API server only
make test      # Run tests
make lint      # Check code style
make format    # Auto-format code
```

## Models

Default model: **LFM2.5 1.2B** (0.5GB, excellent quality)

Available models:
| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| lfm2.5-1.2b | 0.5GB | Excellent | Default, fast |
| lfm2.5-1.2b-8bit | 0.7GB | Excellent | Higher precision |
| qwen3-1.7b | 1.2GB | Excellent | More capable |
| qwen3-4b | 2.1GB | Best | Maximum quality |

## Privacy

- All processing happens locally on your Mac
- No data sent to external servers
- iMessage database accessed read-only
- Embeddings stored in `~/.jarvis/`

## License

Private - All rights reserved
