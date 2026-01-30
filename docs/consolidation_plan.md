# JARVIS Consolidation Plan

**Date**: 2026-01-30
**Status**: Ready for Execution
**Based on**: [audit_report.md](./audit_report.md)

---

## Overview

This plan consolidates the JARVIS codebase from three parallel implementations into one clean, cohesive repository by:

1. **Keeping** the root codebase as the single source of truth
2. **Deleting** v2/ and v3/ directories (standalone, not imported)
3. **Optionally migrating** valuable features from v2/v3 to root
4. **Cleaning up** dead code and empty directories

---

## Proposed Final Directory Structure

```
jarvis-ai-assistant/
├── README.md                    # Project overview
├── CLAUDE.md                    # Claude Code instructions
├── BENCHMARKS.md               # Benchmark results
├── Makefile                     # Build commands
├── pyproject.toml              # Dependencies
├── .gitignore
│
├── jarvis/                      # Core CLI and business logic
│   ├── __init__.py
│   ├── __main__.py             # Entry point
│   ├── cli.py                  # CLI commands
│   ├── config.py               # Pydantic configuration
│   ├── errors.py               # Exception hierarchy
│   ├── context.py              # RAG context fetcher
│   ├── generation.py           # Health-aware generation
│   ├── prompts.py              # Centralized prompts
│   ├── intent.py               # Intent classification
│   ├── embeddings.py           # Embedding search
│   ├── semantic_search.py      # Semantic message search
│   ├── export.py               # Export (JSON/CSV/TXT)
│   ├── metrics.py              # Prometheus metrics
│   ├── setup.py                # Environment wizard
│   ├── fallbacks.py            # Fallback responses
│   ├── retry.py                # Retry logic
│   ├── api.py                  # AsyncIO utilities
│   ├── api_models.py           # Pydantic schemas
│   ├── threading.py            # Thread-safe handling
│   ├── system.py               # System initialization
│   └── tasks/                  # Async task queue
│
├── api/                         # FastAPI REST layer
│   ├── main.py                 # App entry point
│   ├── dependencies.py         # DI containers
│   ├── errors.py               # HTTP error mapping
│   └── routers/                # 28 API routers
│
├── models/                      # MLX model inference
│   ├── __init__.py             # Singleton exports
│   ├── loader.py               # MLXModelLoader
│   ├── generator.py            # Generation orchestrator
│   ├── registry.py             # Model registry (0.5B/1.5B/3B)
│   ├── templates.py            # 25+ response templates
│   └── prompt_builder.py       # Prompt construction
│
├── core/                        # Infrastructure
│   ├── memory/                 # Memory controller
│   │   ├── controller.py       # 3-tier modes
│   │   └── monitor.py          # Real-time profiling
│   └── health/                 # Health monitoring
│       ├── degradation.py      # Circuit breaker
│       ├── permissions.py      # macOS permissions
│       ├── schema.py           # iMessage schema
│       └── circuit.py          # Base circuit breaker
│
├── integrations/                # External systems
│   ├── imessage/               # iMessage database
│   │   ├── reader.py           # ChatDBReader
│   │   ├── queries.py          # SQL + schema detection
│   │   ├── parser.py           # Message parsing
│   │   ├── avatar.py           # Contact avatars
│   │   └── sender.py           # DEPRECATED
│   └── calendar/               # macOS Calendar
│       ├── reader.py
│       ├── writer.py
│       └── detector.py
│
├── contracts/                   # Protocol definitions
│   ├── memory.py
│   ├── hallucination.py
│   ├── latency.py
│   ├── health.py
│   ├── models.py
│   ├── imessage.py
│   └── calendar.py
│
├── benchmarks/                  # Validation gates
│   ├── memory/                 # G1: Memory benchmark
│   ├── hallucination/          # G2: HHEM benchmark
│   ├── latency/                # G3/G4: Latency benchmarks
│   └── templates/              # Template mining
│
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── scripts/                     # Utility scripts
│   ├── generate_report.py
│   ├── check_gates.py
│   └── overnight_eval.sh
│
├── desktop/                     # Tauri desktop app
│   ├── src/                    # Svelte frontend
│   ├── src-tauri/              # Rust backend
│   └── package.json
│
├── mcp_server/                  # Claude Code integration
│   ├── server.py
│   ├── tools.py
│   └── handlers.py
│
├── docs/                        # Documentation
│   ├── CLI_GUIDE.md
│   ├── API_REFERENCE.md
│   ├── CODEBASE_AUDIT_REPORT.md
│   ├── audit_report.md         # NEW: This audit
│   ├── consolidation_plan.md   # NEW: This plan
│   └── known_issues.md         # NEW: Known issues
│
└── results/                     # Benchmark results
```

---

## Component Selection Matrix

| Component | Source | Rationale |
|-----------|--------|-----------|
| CLI | Root (`jarvis/cli.py`) | Most complete, tested |
| Config | Root (`jarvis/config.py`) | Pydantic, production-ready |
| Errors | Root (`jarvis/errors.py`) | Full hierarchy |
| iMessageReader | Root (`integrations/imessage/reader.py`) | Attachments, reactions, protocols |
| ModelLoader | Root (`models/loader.py`) | Timeouts, memory checks |
| Generator | Root (`models/generator.py`) | Template-first design |
| Templates | Root (`models/templates.py`) | 25+ templates |
| Intent | Root (`jarvis/intent.py`) | Semantic routing |
| Prompts | Root (`jarvis/prompts.py`) | Centralized |
| Memory | Root (`core/memory/`) | 3-tier modes |
| Health | Root (`core/health/`) | Circuit breaker |
| Benchmarks | Root (`benchmarks/`) | G1-G4 gates |
| API | Root (`api/`) | 28 routers |
| Tests | Root (`tests/`) | 60+ files |
| Desktop | Root (`desktop/`) | Tauri + Svelte |
| MCP | Root (`mcp_server/`) | Claude integration |

---

## What to Delete

### Immediate Deletion (No Dependencies)

| Path | Size | Reason |
|------|------|--------|
| `v2/` | ~15,000 LOC | Standalone, not imported |
| `v3/` | ~8,000 LOC | Standalone, not imported |
| `mlx-bitnet/` | 0 files | Empty placeholder |

### Scripts Cleanup (Optional)

| Path | Reason |
|------|--------|
| `v2/scripts/archive/` | Abandoned experiments (within v2) |

---

## Migration Checklist

### Phase 1: Verification (Before Deletion)

- [ ] Run `make verify` - all tests pass
- [ ] Run `make health` - system healthy
- [ ] Confirm v2/v3 have NO imports from root: `grep -r "from v2\|from v3" --include="*.py"`
- [ ] Confirm root has NO imports from v2/v3: `grep -r "import v2\|import v3" --include="*.py"`

### Phase 2: Deletion

- [ ] Delete `v2/` directory
- [ ] Delete `v3/` directory
- [ ] Delete `mlx-bitnet/` directory

### Phase 3: Cleanup

- [ ] Update `.gitignore` if needed
- [ ] Update `README.md` to remove v2/v3 references
- [ ] Update `CLAUDE.md` to remove v2/v3 references
- [ ] Run `make verify` again

### Phase 4: Feature Migration (Optional)

If desired, migrate valuable v2/v3 features:

| Feature | Action | Priority |
|---------|--------|----------|
| StyleAnalyzer | Copy to `jarvis/style.py`, integrate | Optional |
| ContextAnalyzer | Enhance `jarvis/context.py` | Optional |
| RAG Prompts | Add to `jarvis/prompts.py` | Optional |

---

## Interface Contracts

All implementations must satisfy these protocols:

### `contracts/memory.py`
```python
class MemoryProfiler(Protocol):
    def profile_model_stack(self) -> MemoryProfile: ...
    def get_current_usage(self) -> int: ...

class MemoryController(Protocol):
    def get_mode(self) -> MemoryMode: ...
    def get_state(self) -> MemoryState: ...
```

### `contracts/models.py`
```python
class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
    def is_loaded(self) -> bool: ...
    def load(self) -> bool: ...
    def unload(self) -> None: ...
```

### `contracts/imessage.py`
```python
class iMessageReader(Protocol):
    def check_access(self) -> bool: ...
    def get_conversations(self, limit: int = 50) -> list[Conversation]: ...
    def get_messages(self, chat_id: str, limit: int = 100) -> list[Message]: ...
    def search(self, query: str, limit: int = 50) -> list[Message]: ...
```

### `contracts/health.py`
```python
class DegradationController(Protocol):
    def get_state(self) -> CircuitState: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    def can_proceed(self) -> bool: ...
```

---

## Validation After Consolidation

### Must Pass

```bash
# Full verification
make verify

# Individual checks
make test          # All tests pass
make lint          # No lint errors
make typecheck     # No type errors

# Health check
make health
```

### End-to-End Test

```bash
# CLI help
python -m jarvis --help

# Version
python -m jarvis version

# Setup check
python -m jarvis.setup --check

# API server (manual test)
python -m jarvis serve --reload
```

---

## Rollback Plan

If consolidation causes issues:

1. Git reset to pre-consolidation commit
2. Restore v2/ and v3/ directories
3. Investigate specific failure

The consolidation only DELETES directories, so rollback is straightforward via git history.

---

## Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Review this plan | 5 min | User approval |
| Phase 1: Verification | 2 min | Run make verify |
| Phase 2: Deletion | 1 min | rm -rf v2 v3 mlx-bitnet |
| Phase 3: Cleanup | 5 min | Update docs |
| Phase 4: Validation | 2 min | Run make verify |
| Total | ~15 min | |

---

## Approval

- [ ] User approves consolidation plan
- [ ] Proceed with execution
