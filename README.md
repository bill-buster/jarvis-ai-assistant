# Jarvis AI Assistant

Local-first AI assistant for macOS with email and iMessage integration.

## Project Status

This project is under active development. See the implementation status below.

| Component | Status |
|-----------|--------|
| Template Coverage Analysis | Complete |
| MLX Model Generator | Complete |
| iMessage Reader | Mostly Complete |
| Memory Benchmarks | Not Started |
| HHEM Benchmarks | Not Started |
| Latency Benchmarks | Not Started |
| Core Infrastructure | Not Started |
| Gmail Integration | Not Started |

## Features

### Implemented

- iMessage integration (read-only, local database access)
- MLX-based language model generation on Apple Silicon
- Template-first generation with semantic matching (75 templates)
- No cloud data transmission - fully local inference

### Planned (Not Yet Implemented)

- Gmail API integration
- Memory-aware model loading
- HHEM hallucination validation
- Graceful degradation system

## Requirements

- macOS on Apple Silicon (M1/M2/M3)
- Python 3.11+
- 8-16GB RAM recommended
- [uv](https://docs.astral.sh/uv/) package manager

## Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd jarvis-ai-assistant
make setup  # Installs deps + enables git hooks

# Verify everything works
make verify
```

## Common Commands

```bash
make test          # Run tests (results in test_results.txt)
make check         # Run all linters
make verify        # Full verification (lint + test)
make health        # Project health summary
make help          # List all available commands
```

## Running Benchmarks

Currently only template coverage analysis is implemented:

```bash
# Run template coverage analysis
python -m benchmarks.coverage.run --output results/coverage.json

# Results show coverage at 0.5, 0.7, and 0.9 similarity thresholds
```

## Workflow

1. Create feature branch: `git checkout -b feature/my-thing`
2. Make changes
3. Run `make verify` before committing
4. Push and create PR

For parallel work, use worktrees - see [CLAUDE.md](CLAUDE.md) for details.

## Project Structure

```
jarvis-ai-assistant/
├── benchmarks/      # Validation gates (only coverage/ is implemented)
│   ├── coverage/    # Template coverage analyzer (COMPLETE)
│   ├── memory/      # Memory profiler (STUB)
│   ├── hallucination/  # HHEM benchmark (STUB)
│   └── latency/     # Latency benchmark (STUB)
├── contracts/       # Python Protocol interfaces (all defined)
├── core/            # Infrastructure (STUBS ONLY)
├── integrations/
│   ├── imessage/    # iMessage reader (MOSTLY COMPLETE)
│   └── gmail/       # Gmail client (STUB)
├── models/          # MLX model loading and inference (COMPLETE)
├── tests/           # Test suite (~536 tests, 98% coverage)
├── scripts/         # Utility scripts
└── docs/            # Design docs and audit report
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development workflow, architecture, and coding guidelines
- [docs/CODEBASE_AUDIT_REPORT.md](docs/CODEBASE_AUDIT_REPORT.md) - Full codebase audit
- [docs/JARVIS-v1-Design-Document.md](docs/JARVIS-v1-Design-Document.md) - Architecture design
- [docs/JARVIS-v1-Development-Guide.md](docs/JARVIS-v1-Development-Guide.md) - Development guide
- [contracts/](contracts/) - Interface definitions and contracts

## License

MIT License - see pyproject.toml
