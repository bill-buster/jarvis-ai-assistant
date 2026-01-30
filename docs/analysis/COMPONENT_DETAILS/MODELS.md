# Models Subsystem Deep Dive

**Last Updated**: 2026-01-27

---

## Overview

The Models subsystem handles text generation using MLX on Apple Silicon with a template-first architecture for fast, hallucination-free responses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLXGenerator                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Template Matcher │  │  Prompt Builder │                   │
│  │  (models/       │  │  (models/       │                   │
│  │   templates.py) │  │   prompt_builder│                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│           ▼                    ▼                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  MLXModelLoader                      │    │
│  │  - Double-check locking for thread safety           │    │
│  │  - Memory pressure checks before load               │    │
│  │  - Metal cache clearing on unload                   │    │
│  │  - Model registry integration                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. MLXModelLoader (`models/loader.py`)

**Purpose**: Thread-safe model lifecycle management

**Key Features**:
- Double-check locking pattern for singleton load
- Memory pressure check before loading
- Explicit Metal GPU cache clearing on unload
- Timeout handling for generation
- Model registry integration for multi-model support

**Memory Management**:
```python
def unload(self) -> None:
    """Unload model and free all memory including GPU cache."""
    self._model = None
    self._tokenizer = None
    mx.metal.clear_cache()  # Critical for Apple Silicon
    gc.collect()
```

### 2. MLXGenerator (`models/generator.py`)

**Purpose**: Orchestrates template matching and model generation

**Generation Flow**:
1. Check template match (similarity >= 0.7)
   - If match: return immediately with `finish_reason="template"`
2. Check memory, load model if needed
3. Build prompt with RAG context + few-shot examples
4. Generate with MLX
5. Return response with metadata

**Thread-Aware Generation**:
The `ThreadAwareGenerator` class provides:
- Topic-specific few-shot examples
- Adjusted temperature per thread type (logistics: 0.3, emotional support: 0.7)
- Response length based on thread type
- Post-processing for thread type

### 3. TemplateMatcher (`models/templates.py`)

**Purpose**: Semantic similarity matching to bypass generation

**Key Features**:
- ~75 built-in templates organized by category
- Custom template support (stored in `~/.jarvis/custom_templates.json`)
- LRU cache for query embeddings (500 max)
- Group size constraints for templates
- Analytics tracking (hit rate, cache efficiency)

**Embedding Model**: all-MiniLM-L6-v2 (~100MB)
**Similarity Threshold**: 0.7 (70% confidence)

### 4. Model Registry (`models/registry.py`)

**Purpose**: Multi-model support for different RAM configurations

**Available Models**:
| ID | Path | Size | Min RAM | Quality |
|----|------|------|---------|---------|
| qwen-0.5b | mlx-community/Qwen2.5-0.5B-Instruct-4bit | 0.8GB | 8GB | basic |
| qwen-1.5b | mlx-community/Qwen2.5-1.5B-Instruct-4bit | 1.5GB | 8GB | good |
| qwen-3b | mlx-community/Qwen2.5-3B-Instruct-4bit | 2.5GB | 16GB | excellent |
| phi3-mini | mlx-community/Phi-3-mini-4k-instruct-4bit | 2.5GB | 8GB | good |
| gemma3-4b | mlx-community/gemma-3-4b-it-4bit | 2.75GB | 8GB | excellent |

**Default**: `qwen-1.5b`

---

## Data Structures

### GenerationRequest
```python
@dataclass
class GenerationRequest:
    prompt: str
    context_documents: list[str]  # RAG context to inject
    few_shot_examples: list[tuple[str, str]]  # (input, output) pairs
    max_tokens: int = 100
    temperature: float = 0.7
    stop_sequences: list[str] | None = None
```

### GenerationResponse
```python
@dataclass
class GenerationResponse:
    text: str
    tokens_used: int
    generation_time_ms: float
    model_name: str
    used_template: bool
    template_name: str | None
    finish_reason: str  # "stop", "length", "template", "fallback", "error"
    error: str | None = None
```

---

## Configuration

Via `jarvis/config.py`:
```python
class ModelConfig:
    model_id: str | None = None
    model_path: str = ""
    estimated_memory_mb: float = 800
    memory_buffer_multiplier: float = 1.1  # 10% safety buffer
    default_max_tokens: int = 100
    default_temperature: float = 0.7
    generation_timeout_seconds: float = 60.0
```

---

## Test Coverage

| File | Coverage | Focus |
|------|----------|-------|
| `test_generator.py` | 99% | Generation, template matching |
| `test_templates.py` | 100% | Template matching, custom templates |
| `test_registry.py` | 100% | Model selection |
| `test_threaded_generation.py` | 100% | Thread-aware generation |

---

## Open Questions

1. **Should default model be gemma3-4b?** Marked as "recommended" in registry
2. **What's the actual memory footprint?** Requires benchmark run
3. **Is 0.7 similarity threshold optimal?** Could experiment

---

## Key Files

- `models/generator.py` (527 lines)
- `models/loader.py` (632 lines)
- `models/templates.py` (2,196 lines)
- `models/registry.py` (270 lines)
- `models/prompt_builder.py` (90 lines)
