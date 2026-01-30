# Benchmarks Subsystem Deep Dive

**Last Updated**: 2026-01-27

---

## Overview

The Benchmarks subsystem implements the validation gates that determine project viability.

---

## Validation Gates

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Total model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G2 | Mean HHEM score | ≥0.5 | 0.4-0.5 | <0.4 |
| G3 | Warm-start latency (p95) | <3s | 3-5s | >5s |
| G4 | Cold-start latency (p95) | <15s | 15-20s | >20s |

---

## 1. Memory Profiler (WS1)

### Purpose
Profile actual memory usage of MLX models during operation.

### Implementation

**File**: `benchmarks/memory/profiler.py` (301 lines)

**Key Features**:
- RSS measurement (actual RAM used)
- Virtual memory tracking
- Metal GPU memory tracking
- Load time measurement
- Automatic model unload after profiling

**Usage**:
```bash
python -m benchmarks.memory.run
```

**Output**:
```json
{
    "model_name": "Qwen2.5-1.5B-Instruct-4bit",
    "quantization": "4bit",
    "context_length": 2048,
    "rss_mb": 1542.3,
    "virtual_mb": 8234.5,
    "metal_mb": 1489.2,
    "load_time_seconds": 3.4,
    "timestamp": "2026-01-27T12:00:00"
}
```

### Dashboard

**File**: `benchmarks/memory/dashboard.py` (348 lines)

**Features**:
- ASCII visualization
- Real-time monitoring
- JSON/CSV export

---

## 2. HHEM Benchmark (WS2)

### Purpose
Evaluate hallucination rates using Vectara's HHEM model.

### Implementation

**File**: `benchmarks/hallucination/hhem.py` (246 lines)

**HHEM Model**: `vectara/hallucination_evaluation_model`
- CrossEncoder from sentence-transformers
- Scores from 0 (hallucinated) to 1 (grounded)
- Target threshold: ≥0.5

**Batch Processing**:
```python
_BATCH_SIZE = 16  # For efficient evaluation
```

**Usage**:
```bash
python -m benchmarks.hallucination.run
```

**Output**:
```json
{
    "model_name": "Qwen2.5-1.5B-Instruct-4bit",
    "num_samples": 100,
    "mean_score": 0.67,
    "median_score": 0.71,
    "std_score": 0.15,
    "pass_rate_at_05": 0.82,
    "pass_rate_at_07": 0.65,
    "timestamp": "2026-01-27T12:00:00"
}
```

### Datasets

**File**: `benchmarks/hallucination/datasets.py` (868 lines)

- RAGTruth-inspired test cases
- Source + summary pairs for evaluation

---

## 3. Latency Benchmark (WS4)

### Purpose
Measure end-to-end latency for different scenarios.

### Implementation

**File**: `benchmarks/latency/run.py` (508 lines)

**Scenarios**:
| Scenario | Description |
|----------|-------------|
| `cold` | Model not loaded, full startup required |
| `warm` | Model loaded, prompt processing only |
| `hot` | Model loaded, cache warmed |

**Measurements**:
- Load time (cold start only)
- Prefill time (prompt processing)
- Generation time
- Total end-to-end time
- Tokens per second

**Usage**:
```bash
python -m benchmarks.latency.run
```

**Output**:
```json
{
    "scenario": "warm",
    "model_name": "Qwen2.5-1.5B-Instruct-4bit",
    "num_runs": 10,
    "p50_ms": 1234.5,
    "p95_ms": 1567.8,
    "p99_ms": 1789.2,
    "mean_ms": 1298.4,
    "std_ms": 156.3,
    "timestamp": "2026-01-27T12:00:00"
}
```

### Timer

**File**: `benchmarks/latency/timer.py` (162 lines)

High-precision timing with `time.perf_counter()`.

---

## 4. Template Mining (Removed WS3)

**Status**: Functionality moved to `models/templates.py`

The original template coverage benchmark was removed. Template matching is now part of the core models subsystem.

---

## Running Benchmarks

### Individual Benchmarks

```bash
python -m benchmarks.memory.run
python -m benchmarks.hallucination.run
python -m benchmarks.latency.run
```

### Overnight Evaluation

```bash
# Full evaluation (all benchmarks)
./scripts/overnight_eval.sh

# Quick mode (reduced iterations)
./scripts/overnight_eval.sh --quick

# Check gate status
python scripts/check_gates.py results/latest
```

### Output Structure

```
results/YYYYMMDD_HHMMSS/
├── eval.log
├── memory.json
├── hhem.json
├── latency.json
└── REPORT.md
```

---

## Gate Evaluation Logic

**File**: `scripts/check_gates.py` (132 lines)

```python
def evaluate_gates(results_dir: Path) -> dict:
    """Evaluate all gates from benchmark results."""
    # G1: Memory
    if memory_mb < 5500:
        g1_status = "PASS"
    elif memory_mb < 6500:
        g1_status = "CONDITIONAL"
    else:
        g1_status = "FAIL"

    # G2: HHEM
    if mean_hhem >= 0.5:
        g2_status = "PASS"
    elif mean_hhem >= 0.4:
        g2_status = "CONDITIONAL"
    else:
        g2_status = "FAIL"

    # G3: Warm latency
    if warm_p95_ms < 3000:
        g3_status = "PASS"
    elif warm_p95_ms < 5000:
        g3_status = "CONDITIONAL"
    else:
        g3_status = "FAIL"

    # G4: Cold latency
    if cold_p95_ms < 15000:
        g4_status = "PASS"
    elif cold_p95_ms < 20000:
        g4_status = "CONDITIONAL"
    else:
        g4_status = "FAIL"
```

---

## Test Coverage

| File | Coverage | Notes |
|------|----------|-------|
| `test_memory_profiler.py` | 99% | Mock-based, no real MLX |
| `test_hhem.py` | 100% | Mock HHEM model |
| `test_latency.py` | 99% | Mock MLX generation |

---

## Key Files

- `benchmarks/memory/profiler.py` (301 lines)
- `benchmarks/memory/dashboard.py` (348 lines)
- `benchmarks/hallucination/hhem.py` (246 lines)
- `benchmarks/hallucination/datasets.py` (868 lines)
- `benchmarks/latency/run.py` (508 lines)
- `benchmarks/latency/timer.py` (162 lines)
- `scripts/overnight_eval.sh` (298 lines)
- `scripts/check_gates.py` (132 lines)
