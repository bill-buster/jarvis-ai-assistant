# Component Dependency Graph

**Last Updated**: 2026-01-27

---

## High-Level Architecture

```
                           ┌─────────────────────────────────┐
                           │          User Interface         │
                           │   ┌─────────┐  ┌─────────────┐  │
                           │   │   CLI   │  │   Desktop   │  │
                           │   │ jarvis/ │  │   Tauri/    │  │
                           │   │ cli.py  │  │   Svelte    │  │
                           │   └────┬────┘  └──────┬──────┘  │
                           └────────┼──────────────┼─────────┘
                                    │              │
                           ┌────────▼──────────────▼─────────┐
                           │          API Layer              │
                           │   ┌──────────────────────────┐  │
                           │   │   FastAPI (29 routers)   │  │
                           │   │   api/main.py            │  │
                           │   └────────────┬─────────────┘  │
                           └────────────────┼────────────────┘
                                            │
         ┌──────────────────────────────────┼────────────────────────────────────┐
         │                                  │                                    │
         ▼                                  ▼                                    ▼
┌─────────────────────┐       ┌──────────────────────────┐       ┌────────────────────────┐
│    iMessage         │       │       Models             │       │    Core Services       │
│    Integration      │       │   ┌──────────────────┐   │       │  ┌──────────────────┐  │
│  ┌───────────────┐  │       │   │ Template Matcher │   │       │  │ Memory Controller│  │
│  │ ChatDBReader  │  │       │   └────────┬─────────┘   │       │  └────────┬─────────┘  │
│  │ (reader.py)   │  │       │            │             │       │           │            │
│  └───────┬───────┘  │       │   ┌────────▼─────────┐   │       │  ┌────────▼─────────┐  │
│          │          │       │   │ MLXGenerator     │   │       │  │ Degradation      │  │
│  ┌───────▼───────┐  │       │   └────────┬─────────┘   │       │  │ Controller       │  │
│  │ Parser        │  │       │            │             │       │  └────────┬─────────┘  │
│  │ (parser.py)   │  │       │   ┌────────▼─────────┐   │       │           │            │
│  └───────────────┘  │       │   │ MLXModelLoader   │   │       │  ┌────────▼─────────┐  │
│          │          │       │   └──────────────────┘   │       │  │ Permission       │  │
│  ┌───────▼───────┐  │       │            │             │       │  │ Monitor          │  │
│  │ Queries       │  │       │   ┌────────▼─────────┐   │       │  └────────┬─────────┘  │
│  │ (queries.py)  │  │       │   │ Model Registry   │   │       │           │            │
│  └───────────────┘  │       │   └──────────────────┘   │       │  ┌────────▼─────────┐  │
└─────────────────────┘       └──────────────────────────┘       │  │ Schema Detector  │  │
                                                                 │  └──────────────────┘  │
                                                                 └────────────────────────┘
```

---

## Dependency Matrix

### Who Depends on What

| Component | Depends On |
|-----------|------------|
| CLI (`jarvis/cli.py`) | API Models, Config, Intent, Context, Generation, Export |
| API Layer (`api/`) | Schemas, Errors, jarvis modules, integrations |
| MLXGenerator | TemplateMatcher, PromptBuilder, MLXModelLoader |
| MLXModelLoader | Model Registry, psutil, mlx, mlx_lm |
| TemplateMatcher | sentence-transformers, Template Analytics |
| ChatDBReader | Parser, Queries, Avatar |
| Degradation Controller | Circuit Breaker |
| Memory Controller | Memory Monitor (psutil) |
| Schema Detector | Queries (single source of truth) |

### Circular Dependencies

**None detected** - Clean dependency graph

---

## Module-Level Dependencies

### jarvis/ (CLI Layer)

```
jarvis/cli.py
├── jarvis/config.py
├── jarvis/intent.py
├── jarvis/context.py
├── jarvis/generation.py
├── jarvis/export.py
├── jarvis/prompts.py
├── jarvis/errors.py
├── jarvis/metrics.py
└── jarvis/setup.py
```

### api/ (API Layer)

```
api/main.py
├── api/schemas.py (4,596 lines - shared Pydantic models)
├── api/errors.py (exception handlers)
├── api/dependencies.py
├── api/ratelimit.py
└── api/routers/*.py (29 routers)
    ├── health.py
    ├── conversations.py
    ├── drafts.py
    ├── suggestions.py
    ├── search.py
    ├── export.py
    ├── settings.py
    ├── metrics.py
    └── ... (21 more)
```

### models/ (Generation Layer)

```
models/generator.py
├── models/loader.py
│   └── models/registry.py
├── models/templates.py
│   └── jarvis/metrics.py (template analytics)
└── models/prompt_builder.py
    └── jarvis/prompts.py
```

### integrations/ (Data Access Layer)

```
integrations/imessage/reader.py
├── integrations/imessage/parser.py
├── integrations/imessage/queries.py
├── integrations/imessage/avatar.py
└── jarvis/errors.py (unified error types)
```

### core/ (Core Services Layer)

```
core/memory/controller.py
└── core/memory/monitor.py

core/health/degradation.py
└── core/health/circuit.py

core/health/permissions.py (standalone)

core/health/schema.py
└── integrations/imessage/queries.py (delegates detection)
```

---

## External Dependencies

### Python Packages

```
mlx (≥0.22.0)
├── Apple Silicon ML framework
└── Used by: models/loader.py

mlx-lm (≥0.22.0)
├── MLX language model utilities
└── Used by: models/loader.py

sentence-transformers (≥5.0.0)
├── Semantic similarity, HHEM evaluation
└── Used by: models/templates.py, benchmarks/hallucination/

psutil (≥7.0.0)
├── System memory monitoring
└── Used by: core/memory/, models/loader.py

pydantic (≥2.10.0)
├── Data validation
└── Used by: jarvis/config.py, api/schemas.py

fastapi (≥0.125.0)
├── REST API framework
└── Used by: api/

uvicorn (≥0.35.0)
├── ASGI server
└── Used by: api/

rich (≥14.0.0)
├── Terminal formatting
└── Used by: jarvis/cli.py
```

### System Dependencies

```
macOS System
├── chat.db (~/Library/Messages/chat.db)
│   └── Requires: Full Disk Access permission
├── AddressBook (~/Library/Application Support/AddressBook/)
│   └── Requires: Contacts permission (optional)
├── Calendar (via EventKit)
│   └── Requires: Calendar permission (optional)
└── Metal GPU (Apple Silicon)
    └── Required for MLX inference
```

---

## Critical Path for Demo

The minimal path to a working demo:

```
1. Full Disk Access granted
         │
         ▼
2. iMessage Reader can access chat.db
         │
         ▼
3. Template Matcher loads embeddings (~100MB)
         │
         ▼
4. For non-template matches:
   MLX Model loads (~1.5GB for qwen-1.5b)
         │
         ▼
5. CLI or API can serve requests
```

### Failure Points

| Point | Failure Mode | Fallback |
|-------|--------------|----------|
| 1 | Permission denied | Setup wizard guides user |
| 2 | Schema unrecognized | Basic text extraction |
| 3 | Network error on first load | Cache or fail gracefully |
| 4 | Out of memory | MINIMAL mode (templates only) |
| 5 | - | - |

---

## Build Order

For a clean build, modules should be built in this order:

1. **contracts/** - Interface definitions (no dependencies)
2. **jarvis/errors.py** - Error hierarchy (no internal deps)
3. **jarvis/config.py** - Configuration (depends on pydantic)
4. **core/memory/** - Memory management (depends on psutil)
5. **core/health/** - Health monitoring (depends on core/memory)
6. **integrations/** - Data access (depends on jarvis/errors)
7. **models/** - Generation (depends on core, integrations)
8. **jarvis/** - CLI (depends on everything)
9. **api/** - REST API (depends on everything)
10. **benchmarks/** - Validation (depends on models, integrations)
