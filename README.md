# JARVIS AI Assistant

Local-first AI assistant for macOS with intelligent iMessage management using MLX-based language models. Runs entirely on Apple Silicon with no cloud data transmission.

## Features

### Core Capabilities

- **iMessage Integration** - Read-only local database access with schema auto-detection (v14/v15)
- **MLX Model Generation** - Local inference on Apple Silicon with memory-aware loading
- **Template-First Generation** - Semantic matching against iMessage scenario templates with configurable thresholds
- **Intent Classification** - Natural language understanding for reply, summarize, and search intents

### AI-Powered Features

- **AI Reply Suggestions** - Context-aware reply generation using conversation history (RAG)
- **Conversation Summaries** - Generate summaries with key points from message history
- **Smart Quick Replies** - Pattern-based suggestions without model invocation

### Export and Data Management

- **Conversation Export** - Export messages to JSON, CSV, or TXT formats
- **Search with Filters** - Full-text search with date range, sender, and attachment filters
- **Backup Support** - Full conversation backup in JSON format

### Performance and Monitoring

- **Prometheus-Compatible Metrics** - Memory, latency, and request metrics at `/metrics`
- **Routing Metrics (SQLite)** - Per-request routing decisions and latency breakdowns (see `scripts/analyze_routing_metrics.py`)
- **Memory Controller** - Three-tier modes based on available RAM (FULL/LITE/MINIMAL)
- **Graceful Degradation** - Circuit breaker pattern for feature failures
- **HHEM Validation** - Post-generation hallucination scoring via Vectara model

### Desktop Application

- **Tauri Desktop App** - Native macOS app built with Svelte
- **Menu Bar Integration** - Quick access from system menu bar
- **AI Draft Panel** - Generate reply suggestions with keyboard shortcut (Cmd+D)
- **Conversation Summary Modal** - One-click summaries with key points (Cmd+S)

**Default Model**: LFM 2.5 1.2B Instruct (4-bit, `lfm-1.2b`)

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- Full Disk Access permission (for iMessage)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd jarvis-ai-assistant
make setup  # Installs deps + enables git hooks

# Run setup wizard (validates environment)
uv run python -m jarvis.setup

# Verify everything works
make verify

# Start using JARVIS
jarvis chat                      # Interactive chat
jarvis search-messages "dinner"  # Search messages
jarvis reply John                # Generate reply suggestions
jarvis summarize Mom             # Summarize a conversation
jarvis health                    # Check system status
```

## CLI Usage

```bash
# Interactive chat with intent-aware routing
jarvis chat

# Search iMessage conversations with filters
jarvis search-messages "meeting tomorrow"
jarvis search-messages "dinner" --limit 50 --sender "John"
jarvis search-messages "project" --start-date 2024-01-01 --has-attachment

# AI-powered reply suggestions
jarvis reply John                          # Generate suggestions
jarvis reply Sarah -i "accept politely"    # With tone instruction

# Conversation summaries
jarvis summarize Mom                       # Last 50 messages
jarvis summarize Boss -n 100               # Last 100 messages

# Export conversations
jarvis export --chat-id <id>               # Export to JSON
jarvis export --chat-id <id> -f csv        # Export to CSV
jarvis export --chat-id <id> -f txt        # Export to TXT

# System health and monitoring
jarvis health

# Run benchmarks
jarvis benchmark memory
jarvis benchmark latency
jarvis benchmark hhem

# Start API server (for desktop app)
jarvis serve                               # Default: localhost:8000
jarvis serve --port 8742 --reload          # Development mode

# Start MCP server (for Claude Code integration)
jarvis mcp-serve                           # Default: stdio mode
jarvis mcp-serve --transport http          # HTTP mode on port 8765

# Version and help
jarvis --version
jarvis --examples                          # Detailed usage examples
```

## Desktop Application

JARVIS includes a native macOS desktop app built with Tauri and Svelte.

```bash
# Start the Python API backend
make api-dev

# In another terminal, start the desktop app
make desktop-setup    # First time only
cd desktop && npm run tauri dev
```

**Features:**
- Menu bar icon with quick access
- Conversation browser with real-time updates
- AI Draft panel (Cmd+D) for reply suggestions
- Summary modal (Cmd+S) for conversation summaries
- Full keyboard navigation

See [desktop/README.md](desktop/README.md) for detailed setup and E2E testing.

## Development Commands

```bash
make test          # Run tests (results in test_results.txt)
make test-fast     # Stop at first failure
make check         # Run all linters (ruff, mypy)
make verify        # Full verification (lint + test)
make health        # Project health summary
make help          # List all available commands
```

## Running Benchmarks

All benchmarks are implemented and functional:

```bash
# Full overnight evaluation (sequential, memory-safe)
./scripts/overnight_eval.sh

# Quick mode for testing
./scripts/overnight_eval.sh --quick

# Individual benchmarks
uv run python -m benchmarks.memory.run --output results/memory.json
uv run python -m benchmarks.hallucination.run --output results/hhem.json
uv run python -m benchmarks.latency.run --output results/latency.json

# Check gate pass/fail status
uv run python scripts/check_gates.py results/latest/
```

### Validation Gates

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G2 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 |
| G3 | Warm-start latency | <3s | 3-5s | >5s |
| G4 | Cold-start latency | <15s | 15-20s | >20s |

## Project Structure

```
jarvis-ai-assistant/
├── jarvis/             # CLI, config, metrics, export, intent classification
│   ├── cli.py          # CLI entry point with all commands
│   ├── config.py       # Nested configuration with migration
│   ├── export.py       # JSON/CSV/TXT export functionality
│   ├── metrics.py      # Memory sampling, latency histograms, caching
│   └── intent.py       # Intent classification for routing
├── api/                # FastAPI REST layer
│   └── routers/        # Conversations, drafts, metrics, settings
├── benchmarks/         # Validation gate implementations
│   ├── memory/         # MLX memory profiler
│   ├── hallucination/  # HHEM benchmark
│   └── latency/        # Latency benchmark
├── contracts/          # Python Protocol interfaces (9 protocols)
├── core/               # Infrastructure
│   ├── health/         # Circuit breaker, degradation, permissions
│   └── memory/         # Memory controller and monitoring
├── integrations/
│   └── imessage/       # iMessage reader with schema detection
├── models/             # MLX model loading and inference
├── tests/              # Test suite
├── scripts/            # Benchmark and reporting utilities
├── desktop/            # Tauri desktop app (Svelte frontend)
└── docs/               # Documentation
```

## Workflow

1. Create feature branch: `git checkout -b feature/my-thing`
2. Make changes
3. Run `make verify` before committing
4. Push and create PR

For parallel work, use git worktrees - see [CLAUDE.md](CLAUDE.md) for details.

## Documentation

- [docs/GUIDE.md](docs/GUIDE.md) - Canonical documentation index
- [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) - Complete CLI reference with examples
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - REST API documentation
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) - Performance tuning and metrics guide
- [desktop/README.md](desktop/README.md) - Desktop app setup and E2E testing
- [CLAUDE.md](CLAUDE.md) - Development workflow, architecture, and coding guidelines

## License

MIT License - see pyproject.toml
