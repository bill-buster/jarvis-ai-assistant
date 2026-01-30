# JARVIS Post-Consolidation Codebase Evaluation Report

## Executive Summary

After consolidating from **204k LOC to ~24k LOC** (removing v2/, v3/, mlx-bitnet/), the JARVIS codebase is well-structured with clear separation of concerns. The codebase is **production-ready for the current architecture** but requires targeted modifications to implement the target OFFLINE/RUNTIME pipeline.

---

## 1. File-by-File Directory Audit

### `jarvis/` - CLI Entry Point & Core Logic
| File | Purpose | Runs? | Dependencies | Exports |
|------|---------|-------|--------------|---------|
| `__init__.py` | Package init, version export | ✓ | None | `__version__` |
| `cli.py` | CLI entry point with argparse commands | ✓ | rich, argcomplete | `run()` |
| `config.py` | Pydantic-based config management | ✓ | pydantic | `JarvisConfig`, `get_config()` |
| `intent.py` | Semantic intent classification | ✓ | sentence-transformers | `IntentClassifier` |
| `errors.py` | Hierarchical exception system | ✓ | None | `JarvisError`, subclasses |
| `context.py` | RAG context fetching | ✓ | integrations.imessage | `ContextFetcher` |
| `prompts.py` | Centralized prompt templates | ✓ | None | `PromptRegistry`, `THREAD_EXAMPLES` |
| `system.py` | System initialization shared by CLI/API | ✓ | core.health, core.memory | `initialize_system()` |
| `export.py` | JSON/CSV/TXT export | ✓ | None | `export_messages()` |
| `metrics.py` | Performance tracking | ✓ | threading | `get_latency_histogram()` |
| `setup.py` | First-time setup wizard | ✓ | All modules | `SetupWizard` |
| `threading.py` | Thread analysis for conversations | ✓ | None | `ThreadAnalyzer` |

### `models/` - MLX Model Infrastructure
| File | Purpose | Runs? | Dependencies | Exports |
|------|---------|-------|--------------|---------|
| `__init__.py` | Singleton generator management | ✓ | mlx-lm | `get_generator()`, `MLXGenerator` |
| `loader.py` (633 lines) | MLX model lifecycle management | ✓* | mlx, mlx-lm | `MLXModelLoader`, `ModelConfig` |
| `generator.py` (534 lines) | Generation orchestration | ✓* | mlx-lm | `MLXGenerator`, `ThreadAwareGenerator` |
| `templates.py` (2197 lines) | Template matching with embeddings | ✓ | sentence-transformers | `TemplateMatcher`, ~100 built-in templates |
| `prompt_builder.py` | RAG-aware prompt construction | ✓ | None | `PromptBuilder` |
| `registry.py` | Multi-model registry (0.5B/1.5B/3B) | ✓ | None | `MODEL_REGISTRY`, `get_recommended_model()` |

*Requires Apple Silicon for MLX operations

### `integrations/imessage/` - iMessage Database Access
| File | Purpose | Runs? | Dependencies | Exports |
|------|---------|-------|--------------|---------|
| `__init__.py` | Package exports | ✓ | None | `ChatDBReader`, `IMessageSender` |
| `reader.py` (1348 lines) | Read-only chat.db access | ✓ | sqlite3 | `ChatDBReader`, `CHAT_DB_PATH` |
| `queries.py` (417 lines) | SQL query templates | ✓ | None | `get_query()`, `detect_schema_version()` |
| `parser.py` (544 lines) | Message parsing utilities | ✓ | plistlib | `parse_attributed_body()`, `normalize_phone_number()` |
| `sender.py` (228 lines) | AppleScript message sending | ⚠️ | subprocess | `IMessageSender` (deprecated) |
| `avatar.py` (216 lines) | Contact avatar retrieval | ✓ | sqlite3 | `get_contact_avatar()` |

### `api/` - FastAPI REST Layer
| File | Purpose | Runs? | Dependencies | Exports |
|------|---------|-------|--------------|---------|
| `main.py` (389 lines) | FastAPI app with 26 routers | ✓ | fastapi, slowapi | `app` |
| `schemas.py` | Pydantic request/response models | ✓ | pydantic | Various schemas |
| `errors.py` | HTTP error mapping | ✓ | fastapi | `register_exception_handlers()` |
| `ratelimit.py` | Rate limiting config | ✓ | slowapi | `limiter` |
| `dependencies.py` | FastAPI dependencies | ✓ | fastapi | Dependency functions |
| **Routers (26 total):** |||
| `routers/conversations.py` | Conversation CRUD | ✓ | integrations.imessage | Router |
| `routers/drafts.py` | AI-powered reply generation | ✓ | models | Router |
| `routers/suggestions.py` | Pattern-based quick replies | ✓ | models.templates | Router |
| `routers/websocket.py` | WebSocket streaming | ✓ | fastapi | Router |
| `routers/embeddings.py` | Semantic search endpoints | ✓ | faiss-cpu | Router |
| `routers/health.py` | Health status | ✓ | core.health | Router |

### `core/` - Infrastructure
| File | Purpose | Runs? | Dependencies | Exports |
|------|---------|-------|--------------|---------|
| `memory/controller.py` (287 lines) | 3-tier memory management | ✓ | psutil | `get_memory_controller()` |
| `memory/monitor.py` | System memory monitoring | ✓ | psutil | `MemoryMonitor` |
| `health/degradation.py` (419 lines) | Circuit breaker pattern | ✓ | None | `get_degradation_controller()` |
| `health/circuit.py` | Circuit breaker state machine | ✓ | None | `CircuitBreaker` |
| `health/permissions.py` | macOS permission checking | ✓ | None | `PermissionMonitor` |
| `health/schema.py` | chat.db schema detection | ✓ | sqlite3 | `ChatDBSchemaDetector` |

### `contracts/` - Protocol Interfaces
| File | Protocols Defined | Implementation Status |
|------|-------------------|----------------------|
| `models.py` | `Generator`, `GenerationRequest`, `GenerationResponse` | ✓ `models/generator.py` |
| `imessage.py` | `iMessageReader`, `Message`, `Conversation`, `Attachment` | ✓ `integrations/imessage/reader.py` |
| `memory.py` | `MemoryController`, `MemoryProfiler` | ✓ `core/memory/`, `benchmarks/memory/` |
| `health.py` | `DegradationController`, `PermissionMonitor` | ✓ `core/health/` |
| `hallucination.py` | `HallucinationEvaluator` | ✓ `benchmarks/hallucination/` |
| `latency.py` | `LatencyBenchmarker` | ✓ `benchmarks/latency/` |
| `calendar.py` | `CalendarReader`, `EventDetector` | ✓ API routers |

### `desktop/src/` - Tauri/Svelte Frontend
| Path | Files | Purpose |
|------|-------|---------|
| `App.svelte` | 1 | Main app shell |
| `lib/components/` | 27 Svelte components | UI components (MessageBubble, AIDraftPanel, StreamingMessage, etc.) |
| `lib/stores/` | 6 TypeScript stores | State management (conversations, health, websocket) |
| `lib/api/` | 4 TypeScript files | API client, WebSocket, types |

---

## 2. Working Code Tests

```python
# Test imports (should run without MLX on any platform)
from contracts.models import GenerationRequest, GenerationResponse, Generator  # ✓
from contracts.imessage import Message, Conversation, iMessageReader  # ✓
from jarvis.config import get_config, JarvisConfig  # ✓
from jarvis.errors import JarvisError, ConfigurationError  # ✓
from jarvis.prompts import PromptRegistry  # ✓
from core.memory import get_memory_controller  # ✓
from core.health import get_degradation_controller  # ✓
from models import get_generator, TemplateMatcher  # ✓ (sentence-transformers required)
from integrations.imessage import ChatDBReader  # ✓ (macOS only for actual use)

# Instantiation tests
config = get_config()  # Creates ~/.jarvis/config.json if missing
mem_ctrl = get_memory_controller()  # Singleton
deg_ctrl = get_degradation_controller()  # Singleton
matcher = TemplateMatcher()  # Loads all-MiniLM-L6-v2 model (~100MB)
```

---

## 3. Data Flow Trace

### CLI Flow: `jarvis reply John`
```
jarvis/cli.py:run()
  → jarvis/cli.py:cmd_reply(contact_name="John")
    → integrations/imessage/reader.py:ChatDBReader.get_conversations()
    → integrations/imessage/reader.py:ChatDBReader.get_messages()
    → jarvis/prompts.py:build_reply_prompt()
    → models/__init__.py:get_generator()
    → models/generator.py:MLXGenerator._try_template_match()  # Fast path
      ↳ models/templates.py:TemplateMatcher.match()
    → models/generator.py:MLXGenerator._generate_with_model()  # If no template match
      ↳ models/loader.py:MLXModelLoader.generate_sync()
    → Output to terminal via rich
```

### API Flow: `POST /drafts/reply`
```
api/main.py:app
  → api/routers/drafts.py:generate_reply()
    → integrations/imessage/reader.py:ChatDBReader
    → models/generator.py:MLXGenerator.generate()
    → api/schemas.py:DraftResponse
  → WebSocket: api/routers/websocket.py for streaming
```

---

## 4. Dependencies Analysis (pyproject.toml)

### Core Dependencies (Required)
| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `mlx` | >=0.22.0 | Apple ML framework | Apple Silicon only |
| `mlx-lm` | >=0.22.0 | Language model loading | Apple Silicon only |
| `sentence-transformers` | >=5.0.0 | Template matching embeddings | Uses all-MiniLM-L6-v2 |
| `faiss-cpu` | >=1.9.0 | Vector similarity search | **ALREADY IN DEPS** ✓ |
| `psutil` | >=7.0.0 | Memory monitoring | Cross-platform |
| `pydantic` | >=2.10.0 | Config validation | |
| `fastapi` | >=0.125.0 | REST API | |
| `uvicorn` | >=0.35.0 | ASGI server | |
| `rich` | >=14.0.0 | CLI formatting | |
| `slowapi` | >=0.1.9 | Rate limiting | |
| `reportlab` | >=4.4.0 | PDF export | |

### Missing for Target Architecture
| Package | Purpose | Status |
|---------|---------|--------|
| `bge-small` via MLX | Embedding generation | Need MLX-compatible model loader |
| SQLite for jarvis.db | Metadata storage | Built-in, need schema |

---

## 5. Gap Analysis: Current vs. Target Architecture

### Target: OFFLINE Pipeline
| Requirement | Current Status | Gap |
|-------------|----------------|-----|
| Extract (trigger, response) pairs from chat.db | ✓ `ChatDBReader.get_messages()` returns all messages | Need pair extraction logic |
| Cluster responses → intent groups | ✗ Not implemented | Need clustering algorithm (sklearn) |
| Embed triggers with MLX bge-small | ✗ Uses sentence-transformers (not MLX) | Need MLX-native embedder |
| Store in FAISS index | ✓ `faiss-cpu` in deps | Need integration with templates |
| Store metadata in jarvis.db | ✗ No jarvis.db schema | Need SQLite schema design |

### Target: RUNTIME Pipeline
| Requirement | Current Status | Gap |
|-------------|----------------|-----|
| User clicks chat in Tauri | ✓ `desktop/src/lib/components/ConversationList.svelte` | Working |
| Embed incoming message (MLX bge-small) | ✗ Uses sentence-transformers | Need MLX embedder wrapper |
| FAISS lookup | ✓ api/routers/embeddings.py exists | May need refinement |
| Template/generate/ask decision logic | ✓ Template matching at 0.7 threshold | May need confidence tiers |
| LFM 2.5 1.2B generates | ✗ Uses Qwen models | Need model registry update |
| WebSocket stream | ✓ `api/routers/websocket.py` + `models/generator.py:generate_stream()` | Working |

### Critical Gaps Summary
| Priority | Gap | Effort |
|----------|-----|--------|
| **HIGH** | No OFFLINE pair extraction/clustering pipeline | Medium |
| **HIGH** | Sentence-transformers vs MLX bge-small mismatch | Medium |
| **MEDIUM** | No jarvis.db schema for metadata | Low |
| **MEDIUM** | Model registry lacks LFM 2.5 1.2B | Low |
| **LOW** | FAISS integration exists but may need tuning | Low |

---

## 6. Environment Requirements

### macOS Permissions
- **Full Disk Access**: Required for `~/Library/Messages/chat.db` and `~/Library/Application Support/AddressBook/`
- **Automation** (optional): Only if using deprecated `IMessageSender`

### Configuration Files
| File | Location | Purpose |
|------|----------|---------|
| `config.json` | `~/.jarvis/config.json` | User settings, model selection |
| `custom_templates.json` | `~/.jarvis/custom_templates.json` | User-defined templates |
| **NEW: jarvis.db** | `~/.jarvis/jarvis.db` | Target: FAISS index metadata |

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `JARVIS_MODEL_PATH` | Auto-detect | Override model location |
| `JARVIS_LOG_LEVEL` | INFO | Logging verbosity |
| `JARVIS_API_PORT` | 8742 | FastAPI port |

### System Requirements
- macOS 10.15+ (Catalina)
- Apple Silicon (M1/M2/M3) for MLX
- Python 3.11+
- 8GB RAM minimum (16GB recommended)

---

## 7. Recommendations

### Immediate Actions
1. **Add MLX bge-small wrapper** in `models/embeddings.py`:
   ```python
   class MLXEmbedder:
       def embed(self, text: str) -> np.ndarray:
           # Use mlx-lm or custom MLX embedding model
   ```

2. **Create jarvis.db schema**:
   ```sql
   CREATE TABLE trigger_response_pairs (
       id INTEGER PRIMARY KEY,
       trigger_text TEXT,
       response_text TEXT,
       embedding BLOB,  -- FAISS ID reference
       intent_cluster TEXT,
       source_chat_id TEXT,
       created_at TIMESTAMP
   );
   ```

3. **Add LFM 2.5 1.2B to model registry** in `models/registry.py`

### Architecture Recommendations
- Keep TemplateMatcher for quick responses (low-latency path)
- Add FAISS-based retrieval as secondary path with confidence scoring
- Implement 3-tier response: Template → FAISS match → LLM generate → Ask for help

---

## 8. Summary

| Metric | Value |
|--------|-------|
| Total Python files | ~80 |
| Estimated LOC | ~24,000 |
| Test coverage | Tests in `tests/` directory |
| Contracts implemented | 9/9 (100%) |
| API endpoints | 26 routers, 100+ endpoints |
| Desktop components | 27 Svelte components |

**Verdict**: The consolidated codebase is clean, well-architected, and follows good patterns (contracts, singletons, circuit breakers). The main work for the target architecture is:
1. Adding OFFLINE extraction pipeline (~500 LOC)
2. Swapping sentence-transformers → MLX embedder (~200 LOC)
3. Creating jarvis.db schema (~50 lines SQL)
4. Adding LFM 2.5 1.2B to registry (~20 lines)
