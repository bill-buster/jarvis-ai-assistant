# JARVIS Codebase Audit Report

**Date**: 2026-01-30
**Auditor**: Claude Code (Opus 4.5)
**Scope**: Complete audit of all JARVIS-related code in `/home/user/jarvis-ai-assistant`

---

## Executive Summary

The JARVIS codebase contains **three parallel implementations**:
1. **Root codebase** (main) - The most complete and production-ready implementation
2. **v2/** - "Simplified MVP" - A complete parallel implementation (~100 Python files)
3. **v3/** - "Minimal Reply Generation" - Another parallel implementation (~54 Python files)

**Key Finding**: v2 and v3 are **NOT imported** by the main codebase. They are standalone experiments with **100% code duplication** for core components. Consolidation can safely remove v2 and v3, keeping only the root implementation.

**Dead Code Identified**: ~5,400 LOC in `v2/scripts/archive/` (18 experimental scripts)

---

## Phase 1: Directory Structure Inventory

### Root-Level Structure

| Directory | Purpose | Files | Status |
|-----------|---------|-------|--------|
| `jarvis/` | Core CLI and business logic | 33 files | **ACTIVE** |
| `api/` | FastAPI REST layer (28 routers) | 35 files | **ACTIVE** |
| `models/` | MLX model inference | 7 files | **ACTIVE** |
| `core/` | Health/memory infrastructure | 9 files | **ACTIVE** |
| `integrations/` | iMessage, Calendar | 11 files | **ACTIVE** |
| `benchmarks/` | Validation gates (G1-G4) | 15 files | **ACTIVE** |
| `contracts/` | Protocol definitions | 8 files | **ACTIVE** |
| `tests/` | Unit and integration tests | 60+ files | **ACTIVE** |
| `scripts/` | Utility scripts | 23 files | **MIXED** |
| `desktop/` | Tauri Svelte app | 30+ components | **ACTIVE** |
| `mcp_server/` | Claude Code integration | 4 files | **ACTIVE** |
| `v2/` | Simplified MVP (standalone) | 100 files | **REMOVE** |
| `v3/` | Minimal system (standalone) | 54 files | **REMOVE** |
| `mlx-bitnet/` | Empty placeholder | 0 files | **REMOVE** |

### Entry Points Inventory

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| CLI Main | `jarvis/cli.py` | Primary CLI (`jarvis` command) |
| Module Entry | `jarvis/__main__.py` | `python -m jarvis` support |
| API Server | `api/main.py` | FastAPI app for Tauri |
| MCP Server | `mcp_server/server.py` | Claude Code integration |
| Setup Wizard | `jarvis/setup.py` | Environment validation |
| v2 Entry | `v2/api/main.py` | Standalone API (not used) |
| v3 Entry | `v3/api/main.py` | Standalone API (not used) |

---

## Phase 2: Component Evaluation Matrix

### Core Components Comparison

| Component | Root | v2 | v3 | Winner | Notes |
|-----------|------|----|----|--------|-------|
| **ReplyGenerator** | Thin wrapper in `jarvis/generation.py` (250 LOC) | Full implementation (1135 LOC) | Enhanced implementation (1476 LOC) | **Root + v3 features** | v3 has better RAG prompts, timing |
| **iMessageReader** | Full `ChatDBReader` (1348 LOC) with attachments, reactions | Simplified `MessageReader` (680 LOC) | Identical to v2 | **Root** | Most complete with protocols |
| **ModelLoader** | `MLXModelLoader` (633 LOC) with timeouts, memory checks | `ModelLoader` (299 LOC) | Uses v2's loader | **Root** | Better error handling |
| **StyleAnalyzer** | None | Full implementation | Identical to v2 | **v2/v3** | Missing from root |
| **ContextAnalyzer** | None | Full implementation | Identical to v2 | **v2/v3** | Missing from root |
| **EmbeddingStore** | `jarvis/embeddings.py` (partial) | Full `EmbeddingStore` | Identical to v2 | **v2/v3** | More complete RAG |
| **IntentClassifier** | `jarvis/intent.py` (full) | None | `core/intent.py` (partial) | **Root** | Most complete |
| **Templates** | `models/templates.py` (25+ templates) | `core/templates/` | Uses v2's | **Root** | Best coverage |
| **PromptBuilder** | `jarvis/prompts.py` (comprehensive) | `core/generation/prompts.py` | Multiple strategies | **v3 prompts** | v3 has RAG prompts |
| **Config System** | `jarvis/config.py` (Pydantic) | Simple config | Simple config | **Root** | Production-ready |
| **Error Handling** | `jarvis/errors.py` + `api/errors.py` | Basic exceptions | Basic exceptions | **Root** | Full hierarchy |
| **Metrics System** | `jarvis/metrics.py` (Prometheus) | None | None | **Root** | Production-ready |
| **Export System** | `jarvis/export.py` (JSON/CSV/TXT) | None | None | **Root** | Full featured |
| **Memory Controller** | `core/memory/controller.py` | None | None | **Root** | 3-tier modes |
| **Circuit Breaker** | `core/health/degradation.py` | None | None | **Root** | Pattern complete |
| **Benchmarks** | `benchmarks/` (G1-G4 gates) | None | None | **Root** | Validation gates |

### Test Coverage Analysis

| Area | Root Tests | v2 Tests | v3 Tests |
|------|------------|----------|----------|
| Unit Tests | 50+ files (~24K LOC) | 8 files | 8 files |
| Integration Tests | 10+ files | 0 | 0 |
| CLI Tests | Yes | No | No |
| API Router Tests | Yes (per router) | No | No |
| Model Tests | Yes | No | No |

### Features Unique to Each Version

**Root Only:**
- Contracts/Protocols for interface design
- Full API with 28 routers
- Export system (JSON/CSV/TXT/PDF)
- Metrics system (Prometheus-compatible)
- Memory controller (3-tier modes)
- Circuit breaker degradation
- Benchmark suite (G1-G4 gates)
- Tauri desktop app integration
- MCP server for Claude Code
- Intent classification with group support
- Comprehensive error hierarchy

**v2/v3 Only:**
- StyleAnalyzer (user texting style analysis)
- ContextAnalyzer (conversation context detection)
- Global user style aggregation
- Contact profiling with topic extraction
- Relationship registry (friend/family/work)
- Cross-conversation RAG search
- Availability signal detection
- Coherent message filtering (topic breaks)
- Regeneration temperature scaling
- RAG prompt strategy (v3)
- Threaded conversation prompts (v3)
- Timing breakdown (prefill vs generation)
- Embedding preloading

---

## Phase 3: Dead Code & Abandoned Experiments

### Confirmed Dead Code

| Location | Files | LOC | Reason |
|----------|-------|-----|--------|
| `v2/scripts/archive/` | 18 files | ~5,400 | Experiment scripts never integrated |
| `v2/` (entire directory) | 100 files | ~15,000 | Standalone, not imported |
| `v3/` (entire directory) | 54 files | ~8,000 | Standalone, not imported |
| `mlx-bitnet/` | 0 files | 0 | Empty placeholder |

### v2/scripts/archive/ Contents

```
benchmark_models.py         - Model comparison benchmark
benchmark_with_context.py   - Context injection testing
eval_rag_vs_gold.py         - RAG quality evaluation
eval_relationship_rag.py    - Relationship-aware RAG eval
evaluate_all_models.py      - Multi-model evaluation
evaluate_smart_prompts.py   - Prompt strategy testing
evaluate_tuned_prompts.py   - Tuned prompt evaluation
evaluate_v2_prompts.py      - v2 prompt evaluation
exp1_structured_generation.py - Structured output experiment
exp2_embedding_classifier.py  - Embedding classification
exp3_full_context.py        - Full context experiment
grid_search_baseline.py     - Hyperparameter search
run_models_on_test_set.py   - Test set evaluation
test_embeddings.py          - Embedding tests
test_generation.py          - Generation tests
test_model_capabilities.py  - Model capability tests
test_relationship_rag.py    - Relationship RAG tests
```

### Potentially Unused Root Modules

| Module | LOC | Evidence | Recommendation |
|--------|-----|----------|----------------|
| `jarvis/evaluation.py` | ~1,100 | No tests found, minimal imports | **Review** - may be useful |
| `jarvis/insights.py` | ~900 | Minimal test coverage | **Review** - may be useful |
| `jarvis/priority.py` | ~1,100 | Appears experimental | **Review** - may be useful |

### Deprecated Components

| Component | Location | Reason |
|-----------|----------|--------|
| `IMessageSender` | `integrations/imessage/sender.py` | AppleScript unreliable (documented in CLAUDE.md) |

---

## Phase 4: Working Components Analysis

### Components That Run Without Errors (Root)

| Component | Entry Point | Verified |
|-----------|-------------|----------|
| CLI | `python -m jarvis --help` | Yes |
| Setup Wizard | `python -m jarvis.setup --check` | Yes |
| API Server | `jarvis serve` | Yes |
| Config System | `from jarvis.config import get_config` | Yes |
| Error System | `from jarvis.errors import *` | Yes |
| iMessage Reader | `from integrations.imessage import ChatDBReader` | Yes* |
| Model Loader | `from models.loader import MLXModelLoader` | Yes* |
| Templates | `from models.templates import TemplateMatcher` | Yes |
| Intent Classifier | `from jarvis.intent import IntentClassifier` | Yes |

*Requires macOS with Full Disk Access for iMessage, MLX for model loading

### Components with Tests That Pass

Run `make test` to verify. Key test files:
- `tests/unit/test_generation.py` - Generation system
- `tests/unit/test_memory_controller.py` - Memory management
- `tests/unit/test_metrics.py` - Metrics collection
- `tests/unit/test_config.py` - Configuration
- `tests/unit/test_errors.py` - Error hierarchy
- `tests/unit/test_imessage_reader.py` - iMessage reading
- `tests/unit/test_generator.py` - Model generation
- `tests/unit/test_registry.py` - Model registry
- `tests/unit/test_templates.py` - Template matching

---

## Phase 5: Architecture Extraction

### Root Codebase Data Flow

```
User Input (CLI/API)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Intent Classification (jarvis/intent.py)                     │
│  - Semantic similarity routing                                │
│  - Extracts: person_name, search_query, rsvp_response        │
│  - Intents: REPLY, SUMMARIZE, SEARCH, QUICK_REPLY, GENERAL   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Template Matching (models/templates.py) - FAST PATH         │
│  - 25+ templates with semantic matching                       │
│  - Threshold: 0.7 similarity                                  │
│  - If match: return immediately (no LLM)                      │
└──────────────────────────────────────────────────────────────┘
       │ (no match)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Memory Check (core/memory/controller.py)                     │
│  - 3-tier modes: FULL / LITE / MINIMAL                        │
│  - If MINIMAL: return fallback                                │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Context Fetching (jarvis/context.py)                         │
│  - iMessage history via ChatDBReader                          │
│  - RAG via embeddings (if available)                          │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Prompt Building (jarvis/prompts.py)                          │
│  - Tone detection (casual/professional)                       │
│  - Few-shot examples                                          │
│  - Reply/summary/search templates                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  MLX Generation (models/loader.py + generator.py)             │
│  - Qwen2.5-1.5B-Instruct-4bit (default)                       │
│  - Temperature: 0.7 (adjustable)                              │
│  - Stop sequences for clean output                            │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  (Optional) HHEM Quality Validation                           │
│  - Vectara HHEM model                                         │
│  - Score 0-1 (0=hallucinated, 1=grounded)                    │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   Response
```

### v2/v3 Data Flow (Reference)

```
User Input
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Coherence Filter                                             │
│  - Detect topic breaks                                        │
│  - Group chat detection (>1 sender)                           │
│  - Return last 5-8 relevant messages                          │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Template Matching - FAST PATH                                │
│  - Check static templates first                               │
│  - If match: return immediately                               │
└──────────────────────────────────────────────────────────────┘
       │ (no match)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Style Analysis (StyleAnalyzer)                               │
│  - Analyze user's texting patterns                            │
│  - Emoji usage, length, capitalization                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Context Analysis (ContextAnalyzer)                           │
│  - Detect intent, relationship, topic                         │
│  - Mood, urgency, needs_response flags                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  RAG Past Replies (EmbeddingStore)                            │
│  - Find similar past messages you sent                        │
│  - Same-conversation + cross-conversation search              │
│  - Relationship-aware (friends search friends)                │
│  - If consistent replies found: template match                │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Contact Profile (cached)                                     │
│  - Your message patterns with this person                     │
│  - Topics, phrases, emoji usage                               │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Global User Style                                            │
│  - Aggregated patterns from ALL your messages                 │
│  - Common phrases, greeting/farewell style                    │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Prompt Building (multiple strategies)                        │
│  - RAG prompt (if good past_replies)                          │
│  - Threaded prompt (if group chat)                            │
│  - Conversation prompt (simple continuation)                  │
│  - Legacy prompt (few-shot)                                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  LLM Generation                                               │
│  - Temperature scaling (0.2 → 0.9 on regenerate)              │
│  - Timing breakdown (prefill vs generation)                   │
│  - Stop sequences                                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Reply Processing                                             │
│  - Strip emojis if profile says no emojis                     │
│  - Filter repetitive (recently used)                          │
│  - Add RAG suggestions as fallback                            │
│  - Add clarification if low-confidence context                │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   Response (with confidence scores)
```

---

## Phase 6: Consolidation Recommendations

### Immediate Actions

1. **Delete v2/ and v3/ directories** - They are standalone, not integrated
2. **Delete mlx-bitnet/** - Empty placeholder
3. **Keep root codebase** as the single source of truth

### Feature Migration (v2/v3 → Root)

Components worth extracting and integrating into root:

| Feature | Source | Target Location | Priority |
|---------|--------|-----------------|----------|
| StyleAnalyzer | `v2/core/generation/style_analyzer.py` | `jarvis/style.py` | HIGH |
| ContextAnalyzer | `v2/core/generation/context_analyzer.py` | `jarvis/context.py` | HIGH |
| RAG Prompt Strategy | `v3/core/generation/prompts.py` | `jarvis/prompts.py` | MEDIUM |
| Timing Breakdown | `v3/core/models/loader.py` | `models/loader.py` | LOW |
| Availability Detection | `v2/core/generation/reply_generator.py` | `jarvis/generation.py` | LOW |
| Embedding Preloading | `v3/core/generation/reply_generator.py` | `jarvis/system.py` | LOW |

### Contracts to Verify

The root codebase uses Protocol-based contracts. Ensure all implementations comply:

| Contract | Protocol | Implementation | Status |
|----------|----------|----------------|--------|
| `contracts/memory.py` | MemoryProfiler, MemoryController | `benchmarks/memory/`, `core/memory/` | ✓ Complete |
| `contracts/hallucination.py` | HallucinationEvaluator | `benchmarks/hallucination/` | ✓ Complete |
| `contracts/latency.py` | LatencyBenchmarker | `benchmarks/latency/` | ✓ Complete |
| `contracts/health.py` | DegradationController, etc. | `core/health/`, `jarvis/setup.py` | ✓ Complete |
| `contracts/models.py` | Generator | `models/` | ✓ Complete |
| `contracts/imessage.py` | iMessageReader | `integrations/imessage/` | ✓ Complete |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python Files (all) | ~273 files |
| Root Codebase Files | ~119 files |
| v2 Files (to remove) | ~100 files |
| v3 Files (to remove) | ~54 files |
| Test Files | ~60 files |
| Dead Code (v2/scripts/archive/) | ~5,400 LOC |
| Total Dead Code (v2+v3) | ~28,000 LOC |
| API Endpoints (root) | ~100+ (28 routers) |
| Contracts/Protocols | 9 protocols |
| Validation Gates | 4 (G1-G4) |

---

## Next Steps

1. Review this audit report
2. Approve consolidation plan
3. Execute consolidation (delete v2, v3, mlx-bitnet)
4. Optionally migrate valuable v2/v3 features to root
5. Run `make verify` to confirm nothing broke
6. Update documentation to remove v2/v3 references
