# Decision Log

**Last Updated**: 2026-01-27

This document tracks decisions that need to be made, are pending, or have been made.

---

## Pending Decisions

### D1: Default Model Selection

**Context**: The model registry has 5 models. `qwen-1.5b` is the coded default, but `gemma3-4b` is marked as "recommended" in comments.

**Options**:
| Option | Pros | Cons |
|--------|------|------|
| Keep qwen-1.5b | Smaller (1.5GB), works on 8GB | May have lower quality |
| Switch to gemma3-4b | Better quality per comments | Larger (2.75GB), may not fit 8GB |
| Auto-select by RAM | Best of both worlds | More complexity |

**Recommendation**: Run HHEM benchmark on both, select based on actual quality scores.

**Blocked by**: Benchmark results needed

---

### D2: Cloud Fallback Implementation

**Context**: Design doc mentions cloud fallback for:
- MINIMAL mode (under 8GB RAM)
- Cold-start latency scenarios
- HHEM rejection when local fails quality

**Options**:
| Option | Pros | Cons |
|--------|------|------|
| Implement with Anthropic API | High quality | Cost, privacy concern |
| Implement with Ollama cloud | Self-hosted option | Still requires server |
| Skip for v1 | Simpler | Reduced functionality on low-RAM |

**Recommendation**: Skip for v1, document as limitation.

**Blocked by**: Product decision needed

---

### D3: API Router Test Coverage

**Context**: 21 of 29 API routers lack dedicated tests. The 97% coverage metric is misleading.

**Options**:
| Option | Effort | Impact |
|--------|--------|--------|
| Add tests for all routers | 2-3 weeks | HIGH |
| Add tests for user-facing only | 1 week | MEDIUM |
| Accept current coverage | None | LOW (technical debt) |

**Recommendation**: Add tests for `conversations`, `search`, `tasks` first (user-facing).

**Blocked by**: Time/priority decision

---

### D4: iMessageSender Deprecation

**Context**: CLAUDE.md marks it as experimental/unreliable. Apple's AppleScript restrictions make it fragile.

**Options**:
| Option | Pros | Cons |
|--------|------|------|
| Remove entirely | Clean codebase | Lose write capability |
| Keep but hidden | Available if needed | Technical debt |
| Invest in fixing | Full functionality | High effort, may break with updates |

**Recommendation**: Keep but mark deprecated, remove in v2.

**Blocked by**: None

---

### D5: Template Count Expansion

**Context**: Currently ~75 templates. Design doc suggests 50-100 is target. Coverage benchmark needed to validate.

**Options**:
| Option | Effort | Impact |
|--------|--------|--------|
| Keep current ~75 | None | Unknown coverage |
| Expand to 100+ | 1-2 weeks | May improve coverage |
| Mine from real queries | 1 week | Data-driven expansion |

**Recommendation**: Run coverage benchmark first, then decide.

**Blocked by**: Coverage benchmark results

---

## Made Decisions

### M1: No Fine-Tuning

**Decision**: Use RAG and few-shot prompting only, no fine-tuning.

**Rationale**: Gekhman et al. (EMNLP 2024) shows fine-tuning on new knowledge increases hallucinations.

**Consequences**: Style matching is less precise but hallucination risk is not increased.

**Status**: IMPLEMENTED (no fine-tuning code in codebase)

---

### M2: Three-Tier Memory Modes

**Decision**: Support FULL/LITE/MINIMAL modes with automatic detection.

**Rationale**: 8GB is aspirational but math doesn't work for full functionality.

**Consequences**: More complex code paths but viable across hardware range.

**Status**: IMPLEMENTED (`core/memory/controller.py`)

---

### M3: Template-First Architecture

**Decision**: Match requests to templates first, generate only when no match.

**Rationale**: Generation is expensive (memory, latency) and risky (hallucination).

**Consequences**: Better latency and quality for common cases. Requires template coverage investment.

**Status**: IMPLEMENTED (`models/templates.py`)

---

### M4: Read-Only iMessage Access

**Decision**: Read-only access with schema detection and fallback.

**Rationale**: chat.db access is fragile; Apple changes schema between releases.

**Consequences**: Cannot write to iMessage (acceptable for v1). Resilient to schema changes.

**Status**: IMPLEMENTED (`integrations/imessage/reader.py`)

---

### M5: MLX Over Ollama

**Decision**: Use MLX framework directly instead of Ollama wrapper.

**Rationale**: Better memory control on Apple Silicon, explicit Metal cache clearing.

**Consequences**: More code complexity but precise memory management.

**Status**: IMPLEMENTED (`models/loader.py`)

---

### M6: Sentence-Transformers for Embeddings

**Decision**: Use all-MiniLM-L6-v2 instead of heavier models like GLiNER.

**Rationale**: ~100MB vs 800MB-1.2GB footprint.

**Consequences**: Sufficient for template matching with much lower memory.

**Status**: IMPLEMENTED (`models/templates.py`)

---

### M7: Unified Error Hierarchy

**Decision**: Create centralized error hierarchy with error codes.

**Rationale**: Consistent error handling across CLI and API.

**Consequences**: All errors map to appropriate HTTP status codes.

**Status**: IMPLEMENTED (`jarvis/errors.py`, `api/errors.py`)

---

### M8: Centralized Prompt Registry

**Decision**: Single source of truth for all prompts in `jarvis/prompts.py`.

**Rationale**: Prevents prompt drift and enables versioning.

**Consequences**: All prompts must be defined in one place.

**Status**: IMPLEMENTED (`jarvis/prompts.py`)

---

## Decision Template

```markdown
### Dxx: [Title]

**Context**: [Why this decision is needed]

**Options**:
| Option | Pros | Cons |
|--------|------|------|
| Option 1 | ... | ... |
| Option 2 | ... | ... |

**Recommendation**: [What we suggest]

**Blocked by**: [What info/decision is needed first]
```
