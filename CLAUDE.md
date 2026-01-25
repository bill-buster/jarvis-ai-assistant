# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JARVIS is a local-first AI assistant for macOS that provides intelligent email and iMessage management using MLX-based language models. It runs entirely on Apple Silicon with no cloud data transmission.

### Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Contracts/Interfaces | COMPLETE | All 9 protocol definitions in `contracts/` |
| Template Coverage (WS3) | COMPLETE | 75 templates, 1000 test scenarios |
| Model Generator (WS8) | COMPLETE | MLX loader, template fallback, RAG support |
| iMessage Reader (WS10) | MOSTLY COMPLETE | Has TODOs for attachments/reactions |
| Memory Profiler (WS1) | NOT STARTED | Contract only, stub implementation |
| HHEM Benchmark (WS2) | NOT STARTED | Contract only, stub implementation |
| Latency Benchmark (WS4) | NOT STARTED | Contract only, stub implementation |
| Core Infrastructure (WS5-7) | NOT STARTED | Contracts only, stub implementations |
| Gmail Integration (WS9) | NOT STARTED | Contract only, stub implementation |

**Default Model**: Qwen2.5-0.5B-Instruct-4bit (configured in `models/loader.py`)

See [docs/CODEBASE_AUDIT_REPORT.md](docs/CODEBASE_AUDIT_REPORT.md) for full audit details.

## Quick Reference

```bash
make setup    # First-time setup (install deps + enable hooks)
make test     # Run tests (ALWAYS use this, never raw pytest)
make verify   # Full verification before PR
make health   # Check project status
make help     # List all commands
```

## Build and Development Commands

All commands go through the Makefile. Never run raw pytest or other tools directly.

```bash
# Setup
make install            # Install dependencies via uv sync
make hooks              # Enable git hooks
make setup              # Full dev setup (install + hooks)

# Testing (all capture output to test_results.txt)
make test               # Run full test suite
make test-fast          # Stop at first failure (--maxfail=1)
make test-verbose       # Extra verbosity (-vvv)
make test-coverage      # With coverage report
make test-file FILE=x   # Run single test file

# Code Quality
make lint               # Run ruff check
make format             # Auto-format code
make format-check       # Check formatting (no changes)
make typecheck          # Run mypy
make check              # All static checks

# Verification
make verify             # Full verification (check + test)
make review             # Codebase summary
make health             # Project health status

# Cleanup
make clean              # Remove generated files
make clean-all          # Clean + remove .venv
```

---

## Behavioral Hooks (MANDATORY)

These are checkpoints that must be followed at every stage of development.

### Before Starting Any Task

**Pre-Task Checklist** - Do these BEFORE writing any code:

1. Run `make health` to understand current project state
2. Run `git status` to confirm clean working directory
3. If on `main`, create a feature branch first:
   ```bash
   git checkout -b feature/descriptive-name
   ```
4. Read relevant existing code before writing new code
5. Understand the existing patterns before adding new ones

### Test Execution Rules (MANDATORY)

**These rules are non-negotiable:**

- **ALWAYS** use `make test` or other Makefile test targets - never raw pytest
- **AFTER** tests run, **ALWAYS** read `test_results.txt` before responding
- **NEVER** summarize test failures from memory or truncated terminal output
- If tests fail, **quote the ACTUAL error message** from `test_results.txt`
- If more than 5 tests fail, run `make test-fast` to get full traceback of first failure

Example workflow:
```bash
make test                    # Run tests
# If failures, read the file:
cat test_results.txt         # Get actual error messages
# Quote specific errors in your response
```

### Before Saying "Done" (Self-Verification)

**Before reporting that a task is complete:**

1. Run `make verify` (not just tests - full verification including lint and typecheck)
2. Read `test_results.txt` and confirm all tests pass
3. If you wrote new code, confirm it has test coverage
4. If you modified existing code, confirm existing tests still pass
5. Run `git diff` and review your own changes for obvious issues
6. Only then report completion with **specific evidence**:
   - "All 47 tests pass"
   - "Lint clean, no type errors"
   - NOT "tests should pass now"

### When Tests Fail or Errors Occur

**STOP - don't immediately try to fix.** Follow this process:

1. Read the **FULL** error from `test_results.txt`
2. Identify the **ROOT CAUSE**, not just the symptom
3. If unclear, run `make test-fast` to isolate first failure
4. Fix **ONE** issue at a time
5. Re-run tests after each fix
6. Never say "tests should pass now" without actually running them

### Before Handing Off to Another Agent or Human

When another agent or human will continue your work:

1. Commit all changes with a descriptive message
2. Run `make verify` and paste the output
3. Document what you did and what's left to do
4. Note any gotchas, edge cases, or things you're unsure about
5. If tests are failing, document which ones and why

### Self-Review Before PR

Before creating a PR, review your own diff:

```bash
git diff main..HEAD
```

Checklist:
- [ ] Are there any debug prints or commented code to remove?
- [ ] Are there any hardcoded values that should be config?
- [ ] Did you add/update tests for your changes?
- [ ] Did you update any relevant documentation?
- [ ] Does the code match the project's existing patterns?

---

## Worktree Workflow

For parallel development tasks, use git worktrees to avoid conflicts between branches.

### Creating a Worktree

```bash
# Always start from latest main
git checkout main
git pull origin main

# Create worktree with a new branch
git worktree add ../jarvis-feature-name -b feature/descriptive-name

# Move into the worktree
cd ../jarvis-feature-name

# Each worktree needs its own virtual environment
make install
```

### Working in a Worktree

- Each worktree is an independent working directory with its own `.venv/`
- **Don't share virtual environments across worktrees**
- Run `make install` in each worktree after creation
- Commits in worktrees automatically update the shared git history
- Run `make verify` before considering work complete

### Before Creating PRs

```bash
# Ensure you're up to date with main
git fetch origin
git rebase origin/main

# Resolve any conflicts, then push
git push -u origin feature/descriptive-name

# Create PR via GitHub
```

### Cleanup After Merge

```bash
# From main worktree
cd /path/to/main/repo
git worktree remove ../jarvis-feature-name
git branch -d feature/descriptive-name
```

### Listing Worktrees

```bash
git worktree list
```

---

## Architecture

### Contract-Based Design

The project uses Python Protocols in `contracts/` to enable parallel development across 10 workstreams. All implementations code against these interfaces:

| Contract | Protocol(s) | Implementation Status |
|----------|-------------|----------------------|
| `contracts/memory.py` | MemoryProfiler, MemoryController | CONTRACTS ONLY |
| `contracts/hallucination.py` | HallucinationEvaluator | CONTRACTS ONLY |
| `contracts/coverage.py` | CoverageAnalyzer | IMPLEMENTED in `benchmarks/coverage/` |
| `contracts/latency.py` | LatencyBenchmarker | CONTRACTS ONLY |
| `contracts/health.py` | DegradationController, PermissionMonitor, SchemaDetector | CONTRACTS ONLY |
| `contracts/models.py` | Generator | IMPLEMENTED in `models/` |
| `contracts/gmail.py` | GmailClient | CONTRACTS ONLY |
| `contracts/imessage.py` | iMessageReader | IMPLEMENTED in `integrations/imessage/` |

### Module Structure

| Directory | Purpose | Status |
|-----------|---------|--------|
| `benchmarks/coverage/` | Template matching analysis | COMPLETE |
| `benchmarks/memory/` | Memory profiling (WS1) | STUB ONLY |
| `benchmarks/hallucination/` | HHEM benchmark (WS2) | STUB ONLY |
| `benchmarks/latency/` | Latency benchmark (WS4) | STUB ONLY |
| `core/memory/` | Memory controller (WS5) | STUB ONLY |
| `core/health/` | Health monitoring (WS6-7) | STUB ONLY |
| `core/config/` | Configuration | STUB ONLY |
| `models/` | MLX model inference (WS8) | COMPLETE |
| `integrations/imessage/` | iMessage reader (WS10) | MOSTLY COMPLETE |
| `integrations/gmail/` | Gmail API (WS9) | STUB ONLY |

### Key Patterns (Implemented)

**Template-First Generation**: Queries are matched against templates (semantic similarity via all-MiniLM-L6-v2) before invoking the model. Threshold: 0.7 similarity.

**Thread-Safe Lazy Initialization**: MLXModelLoader uses double-check locking for singleton model loading. See `models/loader.py`.

**Singleton Generator**: Use `get_generator()` to get the shared instance, `reset_generator()` to reinitialize.

**iMessage Schema Detection**: ChatDBReader detects macOS schema versions (v14/v15) and uses version-specific SQL queries. Database is opened read-only with timeout handling for SQLITE_BUSY.

### Patterns (Planned but NOT Implemented)

**Circuit Breaker Degradation**: DegradationController is defined in contracts but has no implementation.

**Memory Controller**: Three-tier memory modes (FULL/LITE/MINIMAL) are defined in contracts but not implemented.

**HHEM Quality Validation**: Hallucination evaluation is defined in contracts but not implemented.

### Data Flow for Text Generation (Current)

1. Template matching (fast path, no model load) - if match >= 0.7, return immediately
2. ~~Memory check via MemoryController~~ (NOT IMPLEMENTED - loads directly)
3. RAG context injection via PromptBuilder
4. Few-shot prompt formatting via PromptBuilder
5. MLX model generation with temperature control
6. ~~HHEM quality validation~~ (NOT IMPLEMENTED)

---

## Validation Gates

Five gates determine project viability. The gate checker script exists (`scripts/check_gates.py`) but requires benchmark outputs that are not yet generated because WS1, WS2, and WS4 are not implemented.

| Gate | Metric | Pass | Conditional | Fail | Can Run? |
|------|--------|------|-------------|------|----------|
| G1 | Template coverage @ 0.7 | >=60% | 40-60% | <40% | YES - `python -m benchmarks.coverage.run` |
| G2 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB | NO - WS1 not implemented |
| G3 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 | NO - WS2 not implemented |
| G4 | Warm-start latency | <3s | 3-5s | >5s | NO - WS4 not implemented |
| G5 | Cold-start latency | <15s | 15-20s | >20s | NO - WS4 not implemented |

### Missing Scripts

The following scripts are referenced in the design doc but do not exist:
- `scripts/overnight_eval.sh` - Would run all benchmarks sequentially
- `scripts/generate_report.py` - Would create BENCHMARKS.md from results

---

## Code Style

- Line length: 100 characters
- Python 3.11+ with strict type hints (mypy strict mode)
- Linting: ruff with E, F, I, N, W, UP rule sets
- Use Pydantic v2 for validated configuration
- Run `make format` before committing

---

## Key Technical Constraints

- **Memory Budget**: Target 8GB minimum, use sequential model loading
- **Read-Only Database Access**: iMessage chat.db must use `file:...?mode=ro` URI
- **No Fine-Tuning**: Research shows it increases hallucinations - use RAG + few-shot instead
- **Model Unloading**: Always unload models between profiles/benchmarks (`gc.collect()`, `mx.metal.clear_cache()`)
