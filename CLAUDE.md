# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JARVIS is a local-first AI assistant for macOS that provides intelligent email and iMessage management using MLX-based language models. It runs entirely on Apple Silicon with no cloud data transmission, targeting a 3B parameter model on devices with 8-16GB RAM.

## Build and Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync
uv sync --extra dev         # Include dev dependencies
uv sync --extra benchmarks  # Include benchmark dependencies

# Run tests
pytest tests/                                    # All tests
pytest tests/unit/                               # Unit tests only
pytest tests/unit/test_coverage.py -v            # Single test file
pytest tests/unit/test_coverage.py::test_name -v # Single test

# Linting and type checking
ruff check .                # Lint
ruff check --fix .          # Auto-fix lint issues
mypy core/ models/ integrations/ benchmarks/

# Run benchmarks (designed for overnight execution, memory-safe for 8GB)
./scripts/overnight_eval.sh

# Run individual benchmarks
python -m benchmarks.memory.run --output results/memory.json
python -m benchmarks.hallucination.run --output results/hhem.json
python -m benchmarks.coverage.run --output results/coverage.json
python -m benchmarks.latency.run --output results/latency.json

# Check validation gates after benchmarks
python scripts/check_gates.py results/latest
```

## Architecture

### Contract-Based Design
The project uses Python Protocols in `contracts/` to enable parallel development across 10 workstreams. All implementations code against these interfaces:

- `contracts/memory.py` - MemoryProfiler, MemoryController (3-tier modes: FULL/LITE/MINIMAL)
- `contracts/hallucination.py` - HallucinationEvaluator (HHEM scoring)
- `contracts/coverage.py` - CoverageAnalyzer (template matching)
- `contracts/latency.py` - LatencyBenchmarker (cold/warm/hot scenarios)
- `contracts/health.py` - DegradationController, PermissionMonitor, SchemaDetector
- `contracts/models.py` - Generator (GenerationRequest/Response)
- `contracts/gmail.py` - GmailClient, Email
- `contracts/imessage.py` - iMessageReader, Message, Conversation

### Module Structure
- `benchmarks/` - Validation gates (WS1-4): memory, hallucination, coverage, latency
- `core/` - Infrastructure (WS5-7): memory controller, health monitoring, config
- `models/` - Model inference (WS8): MLX loader, generator with template fallback
- `integrations/` - External services (WS9-10): Gmail API, iMessage chat.db reader

### Key Patterns

**Template-First Generation**: Queries are matched against templates (semantic similarity via all-MiniLM-L6-v2) before invoking the model. Threshold: 0.7 similarity.

**Thread-Safe Lazy Initialization**: MLXModelLoader uses double-check locking for singleton model loading. See `models/loader.py`.

**iMessage Schema Detection**: ChatDBReader detects macOS schema versions (v14/v15) and uses version-specific SQL queries. Database is opened read-only with timeout handling for SQLITE_BUSY.

**Circuit Breaker Degradation**: DegradationController implements CLOSED -> OPEN -> HALF_OPEN state machine for graceful failure handling.

### Data Flow for Text Generation
1. Template matching (fast path, no model load) - if match >= 0.7, return immediately
2. Memory check via MemoryController - determine operating mode
3. RAG context injection from Gmail/iMessage
4. Few-shot prompt formatting
5. MLX model generation with temperature control
6. (Planned) HHEM quality validation

## Validation Gates

Five gates determine project viability. Run `scripts/check_gates.py` to evaluate:

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Template coverage @ 0.7 | >=60% | 40-60% | <40% |
| G2 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G3 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 |
| G4 | Warm-start latency | <3s | 3-5s | >5s |
| G5 | Cold-start latency | <15s | 15-20s | >20s |

## Code Style

- Line length: 100 characters
- Python 3.11+ with strict type hints (mypy strict mode)
- Linting: ruff with E, F, I, N, W, UP rule sets
- Use Pydantic v2 for validated configuration

## Key Technical Constraints

- **Memory Budget**: Target 8GB minimum, use sequential model loading
- **Read-Only Database Access**: iMessage chat.db must use `file:...?mode=ro` URI
- **No Fine-Tuning**: Research shows it increases hallucinations - use RAG + few-shot instead
- **Model Unloading**: Always unload models between profiles/benchmarks (`gc.collect()`, `mx.metal.clear_cache()`)

## Test Execution Rules

**IMPORTANT: These rules are mandatory for all test runs.**

- **NEVER** run pytest without capturing output to a file
- Always use `make test` or the equivalent command with output capture:
  ```bash
  uv run pytest --tb=long -v --junit-xml=test_results.xml 2>&1 | tee test_results.txt
  ```
- If tests fail, **ALWAYS** read `test_results.txt` to get full tracebacks before responding
- Never summarize test failures from truncated terminal output - read the file first
- After any test run, report the actual error messages from the file, not a summary
- Use `make test-fail-fast` when debugging to stop at the first failure

### Available Make Targets

```bash
make test           # Run all tests with output capture
make test-verbose   # Run with extra verbosity (-vvv)
make test-fail-fast # Stop at first failure (--maxfail=1)
make lint           # Run linters (ruff, mypy)
make clean          # Remove test artifacts and caches
```

## Worktree Workflow

For parallel development tasks, use git worktrees to avoid conflicts between branches.

### Creating a Worktree

```bash
# Always start from latest main
git checkout main
git pull origin main

# Create worktree with a new branch
git worktree add ../jarvis-feature-name -b feature-branch-name

# Move into the worktree
cd ../jarvis-feature-name

# Each worktree needs its own virtual environment
uv sync
```

### Working in a Worktree

- Each worktree is an independent working directory with its own `.venv/`
- Don't share virtual environments across worktrees
- Run `uv sync` in each worktree after creation
- Commits in worktrees automatically update the shared git history

### Before Creating PRs

```bash
# Ensure you're up to date with main
git fetch origin
git rebase origin/main

# Resolve any conflicts, then push
git push -u origin feature-branch-name
```

### Cleanup After Merge

```bash
# From main worktree, remove the feature worktree
git worktree remove ../jarvis-feature-name

# Optionally delete the branch if merged
git branch -d feature-branch-name
```

### Listing Worktrees

```bash
git worktree list
```
