# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JARVIS is a local-first AI assistant for macOS that provides intelligent iMessage management using MLX-based language models. It runs entirely on Apple Silicon with no cloud data transmission.

### Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Contracts/Interfaces | COMPLETE | 9 protocol definitions in `contracts/` |
| Model Generator (WS8) | COMPLETE | MLX loader, template fallback, RAG support |
| iMessage Reader (WS10) | COMPLETE | Schema detection, attachments, reactions |
| Memory Profiler (WS1) | COMPLETE | MLX memory profiling with model unload |
| HHEM Benchmark (WS2) | COMPLETE | Vectara HHEM model evaluation |
| Template Coverage (WS3) | REMOVED | Functionality moved to `models/templates.py` |
| Latency Benchmark (WS4) | COMPLETE | Cold/warm/hot start scenarios |
| Memory Controller (WS5) | COMPLETE | Three-tier modes (FULL/LITE/MINIMAL) |
| Degradation Controller (WS6) | COMPLETE | Circuit breaker pattern |
| Setup Wizard | COMPLETE | Environment validation, config init, health report |
| CLI Entry Point | COMPLETE | `jarvis/cli.py` with chat, search, reply, summarize, export, serve commands |
| FastAPI Layer | COMPLETE | `api/` module for Tauri frontend integration |
| Config System | COMPLETE | `jarvis/config.py` with nested sections and migration |
| Model Registry | COMPLETE | `models/registry.py` with multi-model support (0.5B/1.5B/3B/LFM tiers) |
| Intent Classification | COMPLETE | `jarvis/intent.py` with semantic similarity routing |
| Metrics System | COMPLETE | `jarvis/metrics.py` and `api/routers/metrics.py` for performance monitoring |
| Export System | COMPLETE | `jarvis/export.py` and `api/routers/export.py` for JSON/CSV/TXT export |
| Error Handling | COMPLETE | `jarvis/errors.py` and `api/errors.py` unified exception hierarchy |
| Prompts Registry | COMPLETE | `jarvis/prompts.py` centralized prompt templates and examples |
| MLX Embeddings | COMPLETE | `models/embeddings.py` with MLXEmbedder class for Apple Silicon |
| Reply Router | COMPLETE | `jarvis/router.py` with template/generate/clarify routing logic |
| FAISS Index | COMPLETE | `jarvis/index.py` for trigger similarity search |
| JARVIS Database | COMPLETE | `jarvis/db.py` with contacts, pairs, clusters, embeddings |
| Train/Test Split | COMPLETE | `jarvis/db.py` with `is_holdout` column and split methods |
| Evaluation Pipeline | COMPLETE | `scripts/eval_pipeline.py` for testing on holdout data |
| Embedding Profiles | COMPLETE | `jarvis/embedding_profile.py` for semantic relationship analysis |
| Pair Quality Scoring | COMPLETE | `scripts/score_pair_quality.py` for coherence analysis |

**Default Model**: LFM-2.5-1.2B-Instruct-4bit (configured in `models/registry.py`)

**Known Issues**: See [docs/EVALUATION_AND_KNOWN_ISSUES.md](docs/EVALUATION_AND_KNOWN_ISSUES.md) for detailed issue tracking.

See `docs/GUIDE.md` for current documentation index.

## Quick Reference

```bash
make setup    # First-time setup (install deps + enable hooks)
make test     # Run tests (ALWAYS use this, never raw pytest)
make verify   # Full verification before PR
make health   # Check project status
make help     # List all commands

# JARVIS Setup Wizard
uv run python -m jarvis.setup          # Run full setup (validates environment, creates config)
uv run python -m jarvis.setup --check  # Just check status, don't modify
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

**Four Core Principles (inspired by Karpathy's LLM coding observations):**

1. **Think Before Coding** - Don't assume, ASK. Present options with tradeoffs.
2. **Simplicity First** - Minimum code. No speculation. No overengineering.
3. **Surgical Changes** - Touch ONLY what's needed. No drive-by refactoring.
4. **Goal-Driven** - Define verifiable success criteria. Loop until verified.

---

### Core Principles (Detailed)

**Think Before Coding** - Don't assume, don't hide confusion, surface tradeoffs:
- State assumptions explicitly - if uncertain, ASK rather than guess
- Present multiple approaches when ambiguity exists (with pros/cons)
- Push back constructively if a simpler solution exists
- Stop and clarify when confused - don't code through uncertainty

**Simplicity First** - Minimum code that solves the problem:
- No features beyond what was requested
- No abstractions for single-use code
- No speculative "flexibility" or error handling for impossible scenarios
- If 200 lines could be 50, rewrite it
- Test: Would a senior engineer call this overcomplicated? If yes, simplify.

**Surgical Changes** - Touch only what you must:
- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style even if you'd do it differently
- Remove only imports/variables/functions YOUR changes orphaned
- Test: Every changed line should trace directly to the user's request

**Goal-Driven** - Clear success criteria enable autonomous work:
- Transform vague → verifiable: "Add validation" → "Tests pass for empty/null/invalid/injection"
- For multi-step: State plan with verification at each step
- Loop until criteria met (don't say "should work" without verifying)

### Shell Command Guidelines

- **Always use `uv run`** for Python commands (e.g., `uv run python -m jarvis.setup`, `uv run pytest`). Never use raw `python` or `.venv/bin/python`.
- **Always use `rm -f` or `rm -rf`** when removing files/directories to avoid interactive prompts that can hang
- Use absolute paths when possible to avoid confusion about current directory

### Before Starting Any Task

**Pre-Task Checklist** - Do these BEFORE writing any code:

1. **Think First** - If requirements are unclear, ask clarifying questions NOW:
   - What are the success criteria? (specific, verifiable)
   - Are there multiple valid approaches? Which is simplest?
   - What assumptions am I making?
   - Can I present options with tradeoffs instead of choosing silently?

2. **State your plan for multi-step tasks** (goal-driven execution):
   ```
   Plan:
   1. [Step] → verify: [specific check]
   2. [Step] → verify: [specific check]
   3. [Step] → verify: [specific check]
   ```

3. Run `make health` to understand current project state
4. Run `git status` to confirm clean working directory
5. If on `main`, create a feature branch first:
   ```bash
   git checkout -b feature/descriptive-name
   ```
6. Read relevant existing code before writing new code
7. Understand the existing patterns before adding new ones

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

**Goal-Driven Completion** - Verify against success criteria, not just "it works":

Transform vague criteria into verifiable goals:
- ❌ "Added validation" → ✅ "Tests pass for: empty input, null, invalid format, SQL injection"
- ❌ "Fixed the bug" → ✅ "Test reproduces bug (fails), fix applied, test now passes"
- ❌ "Refactored X" → ✅ "All tests passed before, refactored, all tests still pass"

**Before reporting completion:**

1. Run `make verify` (not just tests - full verification including lint and typecheck)
2. Read `test_results.txt` and confirm all tests pass
3. Verify against the SPECIFIC success criteria defined at task start
4. If you wrote new code, confirm it has test coverage
5. If you modified existing code, confirm existing tests still pass
6. Run `git diff` and review for surgical changes:
   - Do all changes trace to the user's request?
   - Any drive-by refactoring to remove?
   - Any debug code or comments to clean up?
7. Only then report completion with **specific evidence**:
   - "All 47 tests pass"
   - "Lint clean, no type errors"
   - NOT "tests should pass now"

### When Tests Fail or Errors Occur

**STOP - don't immediately try to fix.** Think first, code second:

1. Read the **FULL** error from `test_results.txt`
2. **Surface confusion** - If you don't understand the error, say so and ask
3. Identify the **ROOT CAUSE**, not just the symptom:
   - Is this a wrong assumption I made?
   - Did I overcomplicate something?
   - Did I change code I shouldn't have touched?
4. If unclear, run `make test-fast` to isolate first failure
5. Fix **ONE** issue at a time (surgical changes only)
6. Re-run tests after each fix
7. Never say "tests should pass now" without actually running them

**Common failure patterns:**
- ❌ Made assumption about API behavior → ✅ Should have checked docs/code first
- ❌ Added "flexibility" that broke existing behavior → ✅ Should have kept it simple
- ❌ Refactored adjacent code "while I was there" → ✅ Should have made surgical changes only

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

**Surgical Changes Checklist:**
- [ ] Every changed line traces to the user's request (no drive-by refactoring)
- [ ] No formatting/style changes to code you didn't modify
- [ ] No "improvements" to adjacent functions or comments
- [ ] Removed only imports/variables/functions YOUR changes orphaned
- [ ] Are there any debug prints or commented code to remove?
- [ ] Are there any hardcoded values that should be config?

**Quality Checklist:**
- [ ] Did you add/update tests for your changes?
- [ ] Did you update any relevant documentation?
- [ ] Does the code match the project's existing patterns?
- [ ] Is this the simplest solution? (no overengineering)

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

## CLI Usage

JARVIS provides a command-line interface for interacting with the assistant. For comprehensive documentation, see [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md).

### Running JARVIS

```bash
# Via entry point (after pip install)
jarvis --help

# Via module (development)
uv run python -m jarvis --help

# Show detailed examples
jarvis --examples
```

### Available Commands

```bash
# Interactive chat mode (with intent-aware routing)
jarvis chat

# Search iMessage conversations
jarvis search-messages "meeting tomorrow"
jarvis search-messages "dinner" --limit 50
jarvis search-messages "project" --sender "John" --start-date 2024-01-01
jarvis search-messages "photo" --has-attachment

# Generate reply suggestions
jarvis reply John
jarvis reply Sarah -i "say yes but ask about timing"

# Summarize conversations
jarvis summarize Mom
jarvis summarize Dad -n 100

# Export conversations
jarvis export --chat-id <id>                    # Export to JSON (default)
jarvis export --chat-id <id> -f csv             # Export to CSV format
jarvis export --chat-id <id> -f txt -o out.txt  # Export to TXT with custom filename
jarvis export --chat-id <id> -l 500             # Limit to 500 messages
jarvis export --chat-id <id> --include-attachments  # Include attachment info (CSV)

# Show system health status
jarvis health

# Run benchmarks
jarvis benchmark memory
jarvis benchmark latency
jarvis benchmark hhem
jarvis benchmark memory --output results.json

# Start API server (for Tauri desktop app)
jarvis serve
jarvis serve --host 0.0.0.0 --port 8080
jarvis serve --reload                           # Enable auto-reload for development

# Start MCP server (for Claude Code integration)
jarvis mcp-serve                                # Default: stdio mode
jarvis mcp-serve --transport http               # HTTP mode on port 8765
jarvis mcp-serve --transport http --port 9000   # Custom port

# Version information
jarvis version
jarvis --version
```

### Global Options

```bash
jarvis -v, --verbose    # Enable debug logging
jarvis --version        # Show version and exit
jarvis --examples       # Show detailed usage examples
```

### Permissions Required

- **iMessage**: Requires Full Disk Access
  - Grant in System Settings > Privacy & Security > Full Disk Access

### Shell Completion

JARVIS supports shell completion via argcomplete:

```bash
# Bash (add to ~/.bashrc)
eval "$(register-python-argcomplete jarvis)"

# Zsh (add to ~/.zshrc)
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete jarvis)"

# Fish (create ~/.config/fish/completions/jarvis.fish)
register-python-argcomplete --shell fish jarvis | source
```

---

## Architecture

### Contract-Based Design

The project uses Python Protocols in `contracts/` to enable parallel development. All implementations code against these interfaces:

| Contract | Protocol(s) | Implementation Status |
|----------|-------------|----------------------|
| `contracts/memory.py` | MemoryProfiler, MemoryController | IMPLEMENTED in `benchmarks/memory/` and `core/memory/` |
| `contracts/hallucination.py` | HallucinationEvaluator | IMPLEMENTED in `benchmarks/hallucination/` |
| `contracts/latency.py` | LatencyBenchmarker | IMPLEMENTED in `benchmarks/latency/` |
| `contracts/health.py` | DegradationController, PermissionMonitor, SchemaDetector | IMPLEMENTED in `core/health/` and `jarvis/setup.py` |
| `contracts/models.py` | Generator | IMPLEMENTED in `models/` |
| `contracts/imessage.py` | iMessageReader | IMPLEMENTED in `integrations/imessage/` |

**Total**: 9 protocols across 6 contract files

### Module Structure

| Directory | Purpose | Status |
|-----------|---------|--------|
| `jarvis/` | CLI entry point, config, errors, metrics, export, prompts, intent classification | COMPLETE |
| `api/` | FastAPI REST layer for Tauri frontend (drafts, suggestions, settings, export, metrics, health) | COMPLETE |
| `benchmarks/memory/` | Memory profiling (WS1) | COMPLETE |
| `benchmarks/hallucination/` | HHEM benchmark (WS2) | COMPLETE |
| `benchmarks/latency/` | Latency benchmark (WS4) | COMPLETE |
| `core/memory/` | Memory controller (WS5) | COMPLETE |
| `core/health/` | Health monitoring (WS6) | COMPLETE (circuit breaker + degradation) |
| `models/` | MLX model inference, registry, templates, prompt builder | COMPLETE |
| `integrations/imessage/` | iMessage reader with filters (WS10) | COMPLETE |
| `desktop/` | Tauri desktop app (Svelte frontend) | COMPLETE |
| `tests/` | Unit and integration tests | COMPLETE |

### Key Patterns (Implemented)

**Two Template Systems**: JARVIS has two separate template matching systems:

1. **Static TemplateMatcher** (`models/templates.py`): Matches queries against ~25 canned response templates using semantic similarity (threshold: 0.70). Used for common patterns like greetings, thank-yous, and confirmations. Supports group chat context via `match_with_context(query, group_size)`.

2. **FAISS ReplyRouter** (`jarvis/router.py`): Matches incoming messages against historical (trigger, response) pairs extracted from the user's iMessage history. Uses configurable thresholds (see below). This is the primary routing system for personalized replies.

**Group Chat Templates**: `models/templates.py` includes 25+ group-specific templates organized into categories:
- Event planning (scheduling, day proposals, conflicts)
- RSVP coordination (yes/no/maybe, +1, headcount)
- Poll responses (option voting, preferences)
- Group logistics (who's bringing what, reservations, carpooling)
- Celebratory messages (birthdays, congrats, holidays)
- Information sharing (FYI, updates, reminders)
Templates can specify `min_group_size` and `max_group_size` constraints for size-appropriate responses.

**Thread-Safe Lazy Initialization**: MLXModelLoader uses double-check locking for singleton model loading. See `models/loader.py`.

**Singleton Generator**: Use `get_generator()` to get the shared instance, `reset_generator()` to reinitialize.

**Model Registry**: `models/registry.py` provides multi-model support with `MODEL_REGISTRY` containing specs for 0.5B/1.5B/3B Qwen models and LFM 2.5 1.2B. Use `get_recommended_model(available_ram_gb)` to select the best model for the user's system. Default model is LFM 2.5 1.2B optimized for conversational use.

**Intent Classification**: `IntentClassifier` in `jarvis/intent.py` routes user queries using semantic similarity. Supports REPLY, SUMMARIZE, SEARCH, QUICK_REPLY, GENERAL, and group-specific intents (GROUP_COORDINATION, GROUP_RSVP, GROUP_CELEBRATION) with extracted parameters (person_name, search_query, rsvp_response, poll_choice, etc.).

**Centralized Prompts**: `jarvis/prompts.py` is the single source of truth for all prompts. Includes `PromptRegistry` for dynamic prompt management, few-shot examples for different tones (casual/professional), and prompt templates for replies, summaries, and search answers.

**Unified Error Handling**: `jarvis/errors.py` provides a hierarchical exception system (JarvisError base class with subclasses for Configuration, Model, iMessage, Validation, and Resource errors). `api/errors.py` maps these to appropriate HTTP status codes.

**Metrics System**: `jarvis/metrics.py` provides thread-safe `MemorySampler`, `RequestCounter`, `LatencyHistogram`, and `TTLCache` classes. Use `get_memory_sampler()`, `get_request_counter()`, `get_latency_histogram()` for singleton access. API exposes Prometheus-compatible metrics at `/metrics`.

**iMessage Schema Detection**: Schema detection is consolidated in `integrations/imessage/queries.py` (`detect_schema_version()`). Both `ChatDBReader` and `ChatDBSchemaDetector` in `core/health/schema.py` delegate to this single source of truth. Supports macOS v14 (Sonoma) and v15 (Sequoia) schema versions. Database is opened read-only with timeout handling for SQLITE_BUSY.

**Circuit Breaker Degradation**: `GracefulDegradationController` in `core/health/degradation.py` implements the circuit breaker pattern with states CLOSED -> OPEN -> HALF_OPEN. Use `get_degradation_controller()` for singleton access.

**Memory Controller**: `DefaultMemoryController` in `core/memory/controller.py` provides three-tier memory modes (FULL/LITE/MINIMAL) based on available system memory. Use `get_memory_controller()` for singleton access.

**HHEM Quality Validation**: `HHEMEvaluator` in `benchmarks/hallucination/hhem.py` uses Vectara's HHEM model for hallucination scoring. Scores range from 0 (hallucinated) to 1 (grounded).

**Setup Wizard**: `SetupWizard` in `jarvis/setup.py` validates the environment and guides first-time setup. Checks: platform, Full Disk Access permission, iMessage database schema, system memory, and model availability. Creates `~/.jarvis/config.json` with default settings.

**Export System**: `jarvis/export.py` provides conversation export in JSON (full data with metadata), CSV (flattened for spreadsheets), and TXT (human-readable) formats. Use `export_messages()`, `export_search_results()`, or `export_backup()` functions.

**MLX Embeddings**: `models/embeddings.py` provides `MLXEmbedder` class for fast embedding computation on Apple Silicon using mlx-embeddings. Thread-safe singleton via `get_mlx_embedder()`. Supports bge-small-en-v1.5 (384 dimensions) with automatic L2 normalization.

**Reply Router**: `jarvis/router.py` implements intelligent routing for reply generation via `ReplyRouter`:
- Template (similarity >= 0.90): Returns cached response instantly from FAISS index
- Generate (0.50-0.90): Uses LLM with similar past exchanges as few-shot examples
- Clarify (< 0.50): Asks for more context when message is vague

Thresholds are configurable via `~/.jarvis/config.json` under the `routing` section:
```json
{
  "routing": {
    "template_threshold": 0.90,
    "context_threshold": 0.70,
    "generate_threshold": 0.50
  }
}
```

**Note**: Overnight evaluation (2026-01-30) recommended lower thresholds for better LLM quality:
- `template_threshold`: 0.65 (vs default 0.90)
- `generate_threshold`: 0.45 (vs default 0.50)
See `docs/PLAN.md` Phase 2 for threshold tuning history.

Use `get_reply_router()` for singleton access. API endpoint: `POST /drafts/smart-reply`.

**FAISS Trigger Index**: `jarvis/index.py` provides `TriggerIndexBuilder` and `TriggerIndexSearcher` for versioned vector search. Indexes trigger texts from extracted pairs. Use `build_index_from_db()` to build, `TriggerIndexSearcher.search_with_pairs()` to query.

**JARVIS Database**: `jarvis/db.py` provides `JarvisDB` for managing:
- Contacts with relationship labels and style notes
- Extracted (trigger, response) pairs from message history
- Intent clusters for grouping similar response patterns
- FAISS vector index metadata and versioning
- Train/test split via `is_holdout` column
Use `get_db()` for singleton access. CLI: `jarvis db init`, `jarvis db extract`, `jarvis db build-index`.

**Train/Test Split**: `jarvis/db.py` provides methods for evaluation:
- `split_train_test(holdout_ratio, min_pairs_per_contact, seed)` - Split by contact (all pairs for a contact go to same set)
- `get_training_pairs(min_quality)` - Get pairs where `is_holdout=False`
- `get_holdout_pairs(min_quality)` - Get pairs where `is_holdout=True`
- `get_split_stats()` - Get counts for training/holdout pairs and contacts
Index building excludes holdout pairs by default (`include_holdout=False`).

**Embedding Profiles**: `jarvis/embedding_profile.py` provides semantic relationship analysis:
- Topic clusters via K-means on message embeddings
- Communication dynamics (style similarity, initiation patterns, topic diversity)
- Response semantic shift measurement
Use `build_embedding_profile()` for single contact, `build_profiles_for_all_contacts()` for batch.
Storage: `~/.jarvis/embedding_profiles/{contact_hash}.json`

**Pair Extraction with Context**: `jarvis/extract.py` extracts turn-based pairs with conversation context:
- Groups consecutive messages from same speaker into turns
- Stores up to 5 previous turns as `context_text` for LLM prompts
- Quality scoring based on response time, length, and content patterns
- Filters reactions, generic responses, and topic shifts

### Data Flow for Text Generation (Current)

1. Intent classification via `IntentClassifier` - route to appropriate handler
2. Message classification via `MessageClassifier` - detect acknowledgments, reactions, context requirements
3. FAISS similarity search against historical triggers - get similarity score
4. Route based on thresholds:
   - Score >= 0.90: Template response from FAISS index
   - Score 0.50-0.90: LLM generation with few-shot examples
   - Score < 0.50: Clarification request or cautious generation
5. Memory check via MemoryController - determine operating mode (FULL/LITE/MINIMAL)
6. Context fetching via `ContextFetcher` for iMessage-related intents
7. Prompt building via `jarvis/prompts.py` with tone detection and few-shot examples
8. MLX model generation with temperature control
9. (Optional) HHEM quality validation post-generation

---

## Validation Gates

Four gates determine project viability. All benchmarks are implemented.

| Gate | Metric | Pass | Conditional | Fail | How to Run |
|------|--------|------|-------------|------|------------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB | `uv run python -m benchmarks.memory.run` |
| G2 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 | `uv run python -m benchmarks.hallucination.run` |
| G3 | Warm-start latency | <3s | 3-5s | >5s | `uv run python -m benchmarks.latency.run` |
| G4 | Cold-start latency | <15s | 15-20s | >20s | `uv run python -m benchmarks.latency.run` |

### Benchmark Scripts

- `scripts/generate_report.py` - Generates BENCHMARKS.md from benchmark results
- `scripts/check_gates.py` - Evaluates gate pass/fail status from results
- `scripts/overnight_eval.sh` - Runs all benchmarks sequentially and generates report

### Evaluation Scripts

```bash
# Train/test split and evaluation pipeline
uv run python -m scripts.eval_pipeline --setup              # Create 80/20 split by contact
uv run python -m scripts.eval_pipeline --rebuild-index      # Rebuild FAISS (training only)
uv run python -m scripts.eval_pipeline --limit 100          # Evaluate 100 holdout pairs
uv run python -m scripts.eval_pipeline --output results.json  # Save detailed results

# Pair quality analysis
uv run python -m scripts.score_pair_quality --analyze       # Show quality distribution
uv run python -m scripts.score_pair_quality --update        # Preview quality updates
uv run python -m scripts.score_pair_quality --update --commit  # Apply quality updates

# Embedding profile building
uv run python -m scripts.build_embedding_profiles --contact "Name"  # Single contact
uv run python -m scripts.build_embedding_profiles --limit 50        # Batch build
```

#### Running the Overnight Evaluation

```bash
# Full evaluation (all benchmarks)
./scripts/overnight_eval.sh

# Quick mode (reduced iterations for testing)
./scripts/overnight_eval.sh --quick
```

**Output:**
- Results directory: `results/YYYYMMDD_HHMMSS/`
- Log file: `results/YYYYMMDD_HHMMSS/eval.log`
- Report: `results/YYYYMMDD_HHMMSS/BENCHMARKS.md`
- JSON results: `memory.json`, `hhem.json`, `latency.json`
- Latest symlink: `results/latest/`

**Exit Codes:**
- `0` - All gates pass
- `1` - One gate failed (reassess)
- `2` - Two or more gates failed (consider project cancellation)

**Notes:**
- Memory and latency benchmarks require MLX (Apple Silicon) and will be skipped if unavailable
- Benchmarks run sequentially to respect 8GB memory budget
- Errors in one benchmark don't stop the others from running

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
- **Multi-Model Support**: Model registry supports 0.5B/1.5B/3B tiers with automatic selection based on available RAM
- **Error Handling**: All JARVIS-specific errors inherit from `JarvisError` in `jarvis/errors.py`; API errors are mapped to appropriate HTTP status codes via `api/errors.py`
- **Prompts Centralization**: All prompts must be defined in `jarvis/prompts.py` - do not create prompts in other modules
- **iMessage Sender Limitations**: `IMessageSender` in `integrations/imessage/sender.py` is deprecated. Apple's AppleScript automation has significant restrictions: requires Automation permission, may be blocked by SIP, requires Messages.app running, and may break in future macOS versions. Consider this experimental and unreliable for production use.
